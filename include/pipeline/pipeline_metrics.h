#pragma once  // 头文件保护宏，防止重复包含

/**
 * @file pipeline_metrics.h
 * @brief 管线性能指标收集器
 *
 * 统一各管线的可观测指标（fps / 处理耗时 / 丢帧 / 队列深度 / 重连次数），
 * 支持线程安全的更新与查询。
 */

// 标准库头文件
#include <atomic>     // std::atomic 原子操作类型，线程安全的指标读写
#include <chrono>     // std::chrono 高精度时钟 (steady_clock 单调递增时钟)
#include <cstdint>    // int64_t / uint64_t 等定宽整数类型
#include <iostream>   // std::cerr 用于指标输出到标准错误流
#include <mutex>      // std::mutex 互斥锁，保护非原子操作的计算
#include <string>     // std::string 字符串类型，用于管线名称标签
#include <utility>    // std::move 移动语义

namespace pipeline {  // 管线命名空间

/**
 * @class RateCounter
 * @brief 滑动窗口帧率计算器
 *
 * 基于时间窗口的 FPS 估算：
 * - tick() 每帧累加计数器
 * - 每秒更新一次 rate_ 值
 * - 线程安全（内部 mutex 保护）
 */
class RateCounter {
public:
    /**
     * @brief 记录一次事件（每次采集/处理帧时调用）
     * @param amount 累加值（默认1，即每次调用+1帧）
     *
     * 每秒自动计算一次 rate_：pending_ / elapsed_seconds
     */
    void tick(uint64_t amount = 1) {
        const auto now = std::chrono::steady_clock::now();  // 获取当前单调时钟时间点
        std::lock_guard<std::mutex> lock(mutex_);           // 加锁保护窗口数据
        pending_ += amount;                                  // 累加等待统计的帧数
        const auto elapsed = std::chrono::duration<double>(now - window_start_).count(); // 窗口经过的秒数
        if (elapsed >= 1.0) {                   // 窗口满1秒时更新 rate_
            rate_ = static_cast<double>(pending_) / elapsed; // 计算每秒帧率
            pending_ = 0;                        // 重置累加器
            window_start_ = now;                 // 重置窗口起点
        }
    }

    /**
     * @brief 获取当前帧率（每秒帧数）
     * @return 最近一个完整窗口计算出的帧率
     */
    double rate() const {
        std::lock_guard<std::mutex> lock(mutex_);  // 加锁读取
        return rate_;
    }

private:
    mutable std::mutex mutex_;                     // 互斥锁（mutable 允许 const 方法加锁）
    uint64_t pending_ = 0;                         // 当前窗口内累计帧数
    double rate_ = 0.0;                            // 最近计算的帧率值
    std::chrono::steady_clock::time_point window_start_ = std::chrono::steady_clock::now(); // 当前窗口起始时间
};

/**
 * @struct PipelineMetrics
 * @brief 单管线的运行时指标
 *
 * 汇总一条管线的完整可观测数据，
 * 支持线程安全的原子更新和 FPS 计算。
 */
struct PipelineMetrics {
    PipelineMetrics() = default;                           // 默认构造函数
    explicit PipelineMetrics(std::string n) : name(std::move(n)) {}  // 带名称的构造函数
    std::string name;                     // 管线名称（如 "visible", "rtsp1", "infrared"）

    // ── 帧统计（原子类型，线程安全读写）─────────
    std::atomic<uint64_t> frames_in{0};        // 采集/接收帧数（从源端收到的总帧数）
    std::atomic<uint64_t> frames_processed{0}; // 成功处理帧数（通过处理流程的帧数）
    std::atomic<uint64_t> frames_dropped{0};   // 丢弃帧数（因队列满等原因丢弃的帧数）
    std::atomic<uint64_t> frames_sent{0};      // 推流成功帧数（成功发送到 RTSP 的帧数）

    // ── 耗时统计（微秒，原子类型）────────────────
    std::atomic<int64_t>  last_capture_us{0};      // 上一帧采集耗时（微秒）
    std::atomic<int64_t>  last_preprocess_us{0};   // 上一帧预处理耗时（RGA 转换）（微秒）
    std::atomic<int64_t>  last_inference_us{0};    // 上一帧推理耗时（NPU 推断）（微秒）
    std::atomic<int64_t>  last_encode_us{0};       // 上一帧编码推流耗时（H.264 编码）（微秒）
    std::atomic<int64_t>  last_total_us{0};        // 上一帧端到端耗时（从采集到推流总时间）（微秒）

    // ── 队列指标 ─────────────────────────
    std::atomic<uint32_t> queue_depth{0};      // 当前队列深度（队列中积压的帧数）
    std::atomic<uint64_t> queue_drops{0};      // 队列丢弃次数（因队列满而丢弃的累计次数）

    // ── 连接指标 ─────────────────────────
    std::atomic<uint32_t> reconnect_count{0};  // 重连次数（RTSP 连接断开后重新连接的累计次数）

    // ── FPS 计算（非原子，mutex 保护）─────────
    /**
     * @brief 记录已处理帧数并更新 FPS
     *
     * 每处理完一帧调用一次，每秒刷新 fps() 返回值。
     */
    void tickFrame() {
        frames_processed.fetch_add(1, std::memory_order_relaxed); // 原子递增已处理帧数
        auto now = std::chrono::steady_clock::now();              // 当前单调时间
        std::lock_guard<std::mutex> lock(fps_mtx_);               // 加锁保护 FPS 计算变量
        fps_frame_count_++;                                       // 累加 FPS 窗口内帧数
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - fps_last_time_).count();                        // 距离上次 FPS 更新的毫秒数
        if (elapsed >= 1000) {                    // 每隔1秒更新一次 FPS 值
            current_fps_ = static_cast<double>(fps_frame_count_) * 1000.0 / elapsed; // 计算每秒帧率
            fps_frame_count_ = 0;                 // 重置窗口帧计数
            fps_last_time_ = now;                 // 重置窗口起点
        }
    }

    /**
     * @brief 获取当前 FPS
     * @return 最近一秒内的平均帧率
     */
    double fps() const {
        std::lock_guard<std::mutex> lock(fps_mtx_);  // 加锁读取
        return current_fps_;
    }

    /// 输出所有指标到 std::cerr（用于定期日志）
    void dump() const {
        std::cerr << "[Metrics:" << name << "] "         // 管线名称标签
                  << "fps=" << fps()                     // 当前帧率
                  << " in=" << frames_in.load()          // 累计接收帧数
                  << " proc=" << frames_processed.load() // 累计处理帧数
                  << " drop=" << frames_dropped.load()   // 累计丢弃帧数
                  << " sent=" << frames_sent.load()      // 累计推流帧数
                  << " q=" << queue_depth.load()         // 当前队列深度
                  << " recon=" << reconnect_count.load() // 累计重连次数
                  << " cap_us=" << last_capture_us.load()   // 最近采集耗时
                  << " pre_us=" << last_preprocess_us.load() // 最近预处理耗时
                  << " infer_us=" << last_inference_us.load() // 最近推理耗时
                  << " enc_us=" << last_encode_us.load()     // 最近编码耗时
                  << " total_us=" << last_total_us.load()     // 最近端到端耗时
                  << std::endl;
    }

private:
    mutable std::mutex fps_mtx_;                     // FPS 计算的互斥锁
    double current_fps_  = 0.0;                      // 当前 FPS 值
    int    fps_frame_count_ = 0;                      // 当前 FPS 窗口内帧计数
    std::chrono::steady_clock::time_point fps_last_time_ = std::chrono::steady_clock::now(); // FPS 窗口起点
};

/**
 * @class ScopedTimer
 * @brief RAII 计时器，离开作用域时自动写入目标 atomic
 *
 * 用法：
 * @code
 *   {
 *       ScopedTimer t(metrics.last_inference_us);  // 构造时开始计时
 *       model->interf(result, false);              // 被计时的代码块
 *   } // 离开作用域自动记录耗时到 last_inference_us
 * @endcode
 *
 * 不可拷贝（non-copyable），避免双重触发。
 */
class ScopedTimer {
public:
    /**
     * @brief 构造函数，开始计时
     * @param target 目标 atomic 变量引用，析构时写入耗时
     */
    explicit ScopedTimer(std::atomic<int64_t>& target)
        : target_(target)                                            // 绑定目标 atomic 引用
        , start_(std::chrono::steady_clock::now())                   // 记录计时起点
    {}

    /**
     * @brief 析构函数，自动计算耗时并写入目标变量
     */
    ~ScopedTimer() {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_).count();      // 计算微秒级耗时
        target_.store(elapsed, std::memory_order_relaxed);           // 原子写入目标变量
    }

    /// 获取到目前为止的耗时（微秒），用于中途查询
    int64_t elapsedUs() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_).count();      // 已过微秒数
    }

    // non-copyable（禁止拷贝，避免双重析构）
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::atomic<int64_t>& target_;                     // 目标 atomic 引用（不拥有所有权）
    std::chrono::steady_clock::time_point start_;      // 计时起始时间点
};

} // namespace pipeline
