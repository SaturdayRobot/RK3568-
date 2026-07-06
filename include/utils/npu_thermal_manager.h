#pragma once

#include <algorithm>
#include <cctype>

/**
 * @file npu_thermal_manager.h
 * @brief NPU 温度监控与降级管理
 *
 * 定期读取 RK3568 thermal zone 温度，
 * 根据温度分级提供降级建议（跳帧率、降分辨率、暂停推理等）。
 *
 * 使用方法：
 * @code
 *   NpuThermalManager thermal;
 *   thermal.start();
 *   // 在推理循环中：
 *   if (thermal.shouldSkipInference()) { continue; }
 *   float factor = thermal.throttleFactor(); // 0.0~1.0
 * @endcode
 *
 * RK3568 默认温控区间：
 *   - < 65°C : 正常
 *   - 65~75°C: 轻度降级（跳帧）
 *   - 75~85°C: 中度降级（降分辨率 + 跳帧）
 *   - > 85°C : 重度降级（暂停推理）
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "config/ini_config.h"
#include "utils/thread_runtime.h"

namespace utils {

/**
 * @enum ThermalLevel
 * @brief NPU 温度等级
 */
enum class ThermalLevel : uint8_t {
    Normal = 0,    ///< 正常运行
    Light  = 1,    ///< 轻度降级：推理跳帧 50%
    Medium = 2,    ///< 中度降级：推理跳帧 75% + 降低分辨率
    Heavy  = 3,    ///< 重度降级：暂停推理
};

/**
 * @struct ThermalConfig
 * @brief 温控配置
 */
struct ThermalConfig {
    bool enable = true;          ///< 是否启用温控监控
    /// sysfs 温度文件路径，由板端配置确定。
    std::string sysfs_path = "/sys/class/thermal/thermal_zone1/temp";
    int poll_interval_ms = 2000;  ///< 温度采样间隔（毫秒）
    int threshold_light  = 65;    ///< 轻度降级温度阈值（°C）
    int threshold_medium = 75;    ///< 中度降级温度阈值（°C）
    int threshold_heavy  = 85;    ///< 重度降级温度阈值（°C）
    int hysteresis       = 3;     ///< 迟滞量（°C），防止频繁切换

    float throttle_light = 0.65f;   ///< 轻度降级节流系数
    float throttle_medium = 0.35f;  ///< 中度降级节流系数
    float throttle_heavy = 0.05f;   ///< 重度降级节流系数
    bool heavy_skip_inference = true; ///< heavy等级时是否直接跳过推理

    // 预测性温控：基于温度变化率(dT/dt)提前降级，防止热过冲
    bool predictive_enable = true;       ///< 是否启用预测性温控
    int predictive_preempt_temp = 55;    ///< 预测性降级启动温度（°C）
    float predictive_slope_warn = 2.0f;  ///< 温度上升速率警告阈值（°C/min）
};

class NpuThermalManager {
public:
    explicit NpuThermalManager(ThermalConfig config = {})
        : config_(std::move(config)) {}

    ~NpuThermalManager() { stop(); }

    static bool loadFromIni(const std::string& path,
                            ThermalConfig& out,
                            const std::string& section = "thermal") {
        IniConfig cfg;
        if (!cfg.load(path)) {
            return false;
        }

        out.enable = cfg.getBool(section, "enable", out.enable);
        out.sysfs_path = cfg.getString(section, "sysfs_path", out.sysfs_path);
        out.poll_interval_ms = cfg.getInt(section, "poll_interval_ms", out.poll_interval_ms);
        out.threshold_light = cfg.getInt(section, "threshold_light", out.threshold_light);
        out.threshold_medium = cfg.getInt(section, "threshold_medium", out.threshold_medium);
        out.threshold_heavy = cfg.getInt(section, "threshold_heavy", out.threshold_heavy);
        out.hysteresis = cfg.getInt(section, "hysteresis", out.hysteresis);

        out.throttle_light = static_cast<float>(
            cfg.getDouble(section, "throttle_light", out.throttle_light));
        out.throttle_medium = static_cast<float>(
            cfg.getDouble(section, "throttle_medium", out.throttle_medium));
        out.throttle_heavy = static_cast<float>(
            cfg.getDouble(section, "throttle_heavy", out.throttle_heavy));
        out.heavy_skip_inference = cfg.getBool(
            section, "heavy_skip_inference", out.heavy_skip_inference);

        out.poll_interval_ms = (out.poll_interval_ms < 200) ? 200 : out.poll_interval_ms;
        out.threshold_light = std::max(1, out.threshold_light);
        out.threshold_medium = std::max(out.threshold_light + 1, out.threshold_medium);
        out.threshold_heavy = std::max(out.threshold_medium + 1, out.threshold_heavy);
        out.hysteresis = std::max(0, out.hysteresis);

        out.throttle_light = clampFactor(out.throttle_light);
        out.throttle_medium = clampFactor(out.throttle_medium);
        out.throttle_heavy = clampFactor(out.throttle_heavy);
        return true;
    }

    bool configure(const ThermalConfig& config) {
        if (running_.load()) {
            return false;
        }
        config_ = config;
        return true;
    }

    const ThermalConfig& config() const { return config_; }

    /// 启动温度监控线程
    bool start() {
        if (running_.load()) return true;

        if (!config_.enable) {
            std::cout << "[NpuThermalManager] Disabled by config" << std::endl;
            return false;
        }

        std::string resolved_path;
        if (!resolveSysfsPath(resolved_path)) {
            std::cerr << "[NpuThermalManager] No valid thermal sysfs found, thermal monitoring disabled"
                      << std::endl;
            return false;
        }
        config_.sysfs_path = resolved_path;

        // 验证 sysfs 路径可读
        std::ifstream f(config_.sysfs_path);
        if (!f.is_open()) {
            std::cerr << "[NpuThermalManager] Cannot open " << config_.sysfs_path
                      << ", thermal monitoring disabled" << std::endl;
            return false;
        }
        f.close();

        running_ = true;
        worker_ = std::thread([this]() { pollLoop(); });
        std::cout << "[NpuThermalManager] Started, polling "
                  << config_.sysfs_path << " every "
                  << config_.poll_interval_ms << "ms" << std::endl;
        return true;
    }

    /// 停止温度监控
    void stop() {
        if (!running_.load()) return;
        running_ = false;
        // P1-6 修复：通知条件变量以立即唤醒 pollLoop
        stop_cv_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    /// 获取当前 NPU 温度（°C）
    int temperature() const { return temperature_.load(std::memory_order_relaxed); }

    /// 获取当前温度等级
    ThermalLevel level() const {
        return static_cast<ThermalLevel>(level_.load(std::memory_order_relaxed));
    }

    /**
     * @brief 获取推理节流因子
     * @return 0.0~1.0, 1.0 表示全速, 0.0 表示暂停
     *
     * pipeline 可用此值决定跳帧策略：
     *   if (rand01() > thermal.throttleFactor()) skip this frame
     */
    float throttleFactor() const {
        float base_factor;
        switch (level()) {
            case ThermalLevel::Normal: base_factor = 1.0f; break;
            case ThermalLevel::Light:  base_factor = clampFactor(config_.throttle_light); break;
            case ThermalLevel::Medium: base_factor = clampFactor(config_.throttle_medium); break;
            case ThermalLevel::Heavy:  base_factor = clampFactor(config_.throttle_heavy); break;
            default: base_factor = 1.0f; break;
        }

        // 预测性温控：如果温度上升速率过快，在达到阈值前提前降级
        if (config_.predictive_enable && level() == ThermalLevel::Normal) {
            const int temp = temperature_.load(std::memory_order_relaxed);
            if (temp >= config_.predictive_preempt_temp) {
                const float slope = computeTempSlope();
                if (slope >= config_.predictive_slope_warn) {
                    // 温度上升速率超过阈值，提前应用轻度降级
                    base_factor = std::min(base_factor, clampFactor(config_.throttle_light));
                }
            }
        }

        return base_factor;
    }

    /// 是否应跳过当前帧推理
    bool shouldSkipInference() const {
        return config_.heavy_skip_inference && level() == ThermalLevel::Heavy;
    }

private:
    static float clampFactor(float value) {
        if (value < 0.0f) return 0.0f;
        if (value > 1.0f) return 1.0f;
        return value;
    }

    static bool isReadableFile(const std::string& path) {
        std::ifstream f(path);
        return f.is_open();
    }

    bool resolveSysfsPath(std::string& out_path) const {
        if (!config_.sysfs_path.empty() && isReadableFile(config_.sysfs_path)) {
            out_path = config_.sysfs_path;
            return true;
        }

        // 自动探测包含 npu 关键词的 thermal zone。
        for (int i = 0; i < 32; ++i) {
            const std::string zone_base = "/sys/class/thermal/thermal_zone" + std::to_string(i);
            std::ifstream type_file(zone_base + "/type");
            if (!type_file.is_open()) {
                continue;
            }

            std::string type;
            std::getline(type_file, type);
            for (char& ch : type) {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
            if (type.find("npu") == std::string::npos) {
                continue;
            }

            const std::string temp_path = zone_base + "/temp";
            if (isReadableFile(temp_path)) {
                out_path = temp_path;
                return true;
            }
        }

        // 兜底候选，兼容不同板卡映射。
        const std::vector<std::string> fallback_paths = {
            "/sys/class/thermal/thermal_zone1/temp",
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/thermal/thermal_zone2/temp",
        };
        for (const auto& path : fallback_paths) {
            if (isReadableFile(path)) {
                out_path = path;
                return true;
            }
        }

        return false;
    }

    void pollLoop() {
        applyThreadRuntime("thermal_monitor", "thermal-monitor");

        std::unique_lock<std::mutex> lock(stop_mtx_);
        while (running_.load()) {
            int temp = readTemperature();
            if (temp >= 0) {
                temperature_.store(temp, std::memory_order_relaxed);

                // 记录温度历史用于预测性温控
                recordTemperature(temp);
                updateLevel(temp);
            }
            //用 cv.wait_for 替代 sleep_for，支持 stop() 立即唤醒
            stop_cv_.wait_for(lock,
                std::chrono::milliseconds(config_.poll_interval_ms),
                [this]() { return !running_.load(); });
        }
    }

    void recordTemperature(int temp) {
        const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        temp_history_[temp_history_idx_] = temp;
        temp_history_ts_[temp_history_idx_] = now_ms;
        temp_history_idx_ = (temp_history_idx_ + 1) % kTempHistorySize;
        if (temp_history_count_ < kTempHistorySize) {
            temp_history_count_++;
        }
    }

    // 计算温度变化率（°C/min），基于最近采样点的线性回归
    float computeTempSlope() const {
        if (temp_history_count_ < 2) {
            return 0.0f;
        }
        const int n = temp_history_count_;
        const int newest_idx = (temp_history_idx_ - 1 + kTempHistorySize) % kTempHistorySize;
        const int oldest_idx = (temp_history_idx_ - n + kTempHistorySize) % kTempHistorySize;

        const float dt_ms = static_cast<float>(temp_history_ts_[newest_idx] - temp_history_ts_[oldest_idx]);
        if (dt_ms <= 0.0f) {
            return 0.0f;
        }
        const float dtemp = static_cast<float>(temp_history_[newest_idx] - temp_history_[oldest_idx]);
        return dtemp / (dt_ms / 60000.0f);  // °C/min
    }

    /**
     * @brief 从 sysfs 读取温度
     * @return 温度（°C），失败返回 -1
     *
     * sysfs 中的温度值通常是毫度（millidegree），需要 /1000
     */
    int readTemperature() {
        std::ifstream f(config_.sysfs_path);
        if (!f.is_open()) return -1;
        int raw = 0;
        f >> raw;
        // Linux thermal sysfs 温度单位为 millidegree Celsius。
        return raw / 1000;
    }

    /**
     * @brief 根据温度更新等级（带迟滞）
     */
    void updateLevel(int temp) {
        ThermalLevel current = level();
        ThermalLevel next = current;

        // 升级（温度上升，无迟滞）
        if (temp >= config_.threshold_heavy) {
            next = ThermalLevel::Heavy;
        } else if (temp >= config_.threshold_medium) {
            next = ThermalLevel::Medium;
        } else if (temp >= config_.threshold_light) {
            next = ThermalLevel::Light;
        }

        // 降级（温度下降，带迟滞防止频繁切换）
        if (next < current) {
            int hyst = config_.hysteresis;
            if (current == ThermalLevel::Heavy && temp > config_.threshold_heavy - hyst) {
                next = current; // 保持
            } else if (current == ThermalLevel::Medium && temp > config_.threshold_medium - hyst) {
                next = current;
            } else if (current == ThermalLevel::Light && temp > config_.threshold_light - hyst) {
                next = current;
            }
        }

        if (next != current) {
            level_.store(static_cast<uint8_t>(next), std::memory_order_relaxed);
            std::cout << "[NpuThermalManager] Level changed: "
                      << static_cast<int>(current) << " → "
                      << static_cast<int>(next)
                      << " (temp=" << temp << "°C)" << std::endl;
        }
    }

    ThermalConfig config_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::atomic<int> temperature_{0};
    std::atomic<uint8_t> level_{0};  // ThermalLevel

    // 用于 stop() 立即唤醒 pollLoop
    std::mutex stop_mtx_;
    std::condition_variable stop_cv_;

    // 预测性温控：温度历史（最近 5 个采样点）
    static constexpr int kTempHistorySize = 5;
    int temp_history_[kTempHistorySize] = {};
    int64_t temp_history_ts_[kTempHistorySize] = {};
    int temp_history_idx_ = 0;
    int temp_history_count_ = 0;
};

} // namespace utils
