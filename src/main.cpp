/**
 * @file main.cpp
 * @brief RK3568巡检机器人边缘计算终端 — 主程序入口
 *
 * 职责：
 *  - 串联各子系统（推理、视频管线、存储、MQTT）的生命周期（初始化 / 运行 / 优雅退出）
 *  - 主循环中以 5s 为周期采集板端状态并输出运行时统计 JSON
 *  - 管理硬件看门狗（/dev/watchdog）防止系统死锁
 *  - 集成 systemd Type=notify 协议（sd_notify）实现服务状态与看门狗上报
 */

// =============================================================================
// C++ 标准库
// =============================================================================
#include <atomic>      // std::atomic<bool> — 无锁线程间停止信号
#include <algorithm>
#include <chrono>      // system_clock / steady_clock / duration_cast — 时间戳与超时
#include <cstdlib>
#include <csignal>     // std::signal — 注册 SIGINT/SIGTERM/SIGPIPE 处理函数
#include <cstring>     // std::strerror — errno 可读化
#include <sstream>     // std::ostringstream — 高效拼接 JSON 统计字符串
#include <thread>      // std::this_thread::sleep_for — 主循环定时休眠

// =============================================================================
// POSIX / Linux 系统头文件
// =============================================================================
#include <fcntl.h>            // open() 的 O_WRONLY 等标志
#include <unistd.h>           // close() / write()
#include <sys/ioctl.h>        // ioctl() — 看门狗配置与喂狗
#include <linux/watchdog.h>   // WDIOC_SETTIMEOUT / WDIOC_KEEPALIVE 等 ioctl 命令码

// systemd sd_notify 集成（Type=notify 服务需要主动上报状态）
// __has_include 是 C++17 编译期特性，用于检测头文件是否存在
#if __has_include(<systemd/sd-daemon.h>)
#include <systemd/sd-daemon.h>   // sd_notify() — 上报 READY/WATCHDOG 状态
#define HAVE_SD_NOTIFY 1
#else
#define HAVE_SD_NOTIFY 0
#endif

// =============================================================================
// 项目内部头文件
// =============================================================================
#include "app/app_initializer.h"   // AppInitializer::initialize() / shutdown()
#include "config/ini_config.h"

namespace { // 匿名命名空间 — 以下符号仅当前翻译单元可见

// -----------------------------------------------------------------------------
// 全局变量
// -----------------------------------------------------------------------------
std::atomic<bool> g_stop{false}; // 全局停止标志 — SIGINT/SIGTERM 触发时置 true，主循环轮询此标志
int g_watchdog_fd = -1;          // 硬件看门狗 fd，-1 = 未打开/不可用

// -----------------------------------------------------------------------------
// 信号处理 — 仅设置原子标志，不在信号上下文中做任何重操作
// -----------------------------------------------------------------------------
void handleSignal(int) {
    g_stop.store(true); // 原子写，通知主循环优雅退出
}

// -----------------------------------------------------------------------------
// 硬件看门狗管理 — /dev/watchdog 需要在超时前周期性写入，否则系统会被硬件复位
// -----------------------------------------------------------------------------

void closeWatchdog();

/** 打开看门狗设备并设置超时时间 */
bool openWatchdog(const char* device, int requested_timeout) {
    g_watchdog_fd = ::open(device, O_WRONLY);
    if (g_watchdog_fd < 0) {
        std::cerr << "[Watchdog] cannot open " << device << ": "
                  << std::strerror(errno) << std::endl;
        return false;
    }
    int timeout = std::max(5, requested_timeout);
    if (::ioctl(g_watchdog_fd, WDIOC_SETTIMEOUT, &timeout) < 0) {
        std::cerr << "[Watchdog] cannot set timeout: " << std::strerror(errno) << std::endl;
        closeWatchdog();
        return false;
    }
    std::cout << "[Watchdog] opened " << device << " timeout=" << timeout << "s" << std::endl;
    return true;
}

/** 喂狗 — 驱动内部倒计时归零，必须在超时前调用（当前每 5s 调用一次） */
void kickWatchdog() {
    if (g_watchdog_fd >= 0) {
        ::ioctl(g_watchdog_fd, WDIOC_KEEPALIVE, nullptr);
    }
}

/** 安全关闭看门狗 — 写入魔术字符 'V' 通知驱动这是正常关闭，避免意外复位 */
void closeWatchdog() {
    if (g_watchdog_fd >= 0) {
        ::write(g_watchdog_fd, "V", 1); // 'V' 字符告知驱动"这是正常关机"
        ::close(g_watchdog_fd);
        g_watchdog_fd = -1;
    }
}

// -----------------------------------------------------------------------------
// 时间工具
// -----------------------------------------------------------------------------

/** 将 time_point 转换为 Unix 毫秒时间戳 */
std::int64_t toUnixMs(std::chrono::system_clock::time_point tp) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               tp.time_since_epoch())
        .count();
}

// -----------------------------------------------------------------------------
// RTSP 管线运行时统计 — 将 VisibleRtspPipeline 指标序列化为 JSON 片段
// -----------------------------------------------------------------------------

/**
 * @param oss     输出 JSON 构建器（引用传递，直接追加）
 * @param name    管线名称（用作 JSON key）
 * @param pipeline 管线指针（nullptr 则跳过）
 * @param first   是否为第一个管道（用于控制 JSON 逗号分隔符），调用后会被置 false
 */
void appendRtspStats(std::ostringstream& oss,
                     const char* name,
                     const pipeline::VisibleRtspPipeline* pipeline,
                     bool& first) {
    if (!pipeline) return;

    if (!first) oss << ',';     // 非首项追加逗号分隔符
    first = false;

    const auto& metrics = pipeline->metrics();
    oss << '"' << name << "\":{";
    oss << "\"fps\":" << metrics.fps() << ',';
    oss << "\"frames_in\":" << metrics.frames_in.load(std::memory_order_relaxed) << ',';
    oss << "\"frames_processed\":" << metrics.frames_processed.load(std::memory_order_relaxed) << ',';
    oss << "\"frames_dropped\":" << metrics.frames_dropped.load(std::memory_order_relaxed) << ',';
    oss << "\"frames_sent\":" << metrics.frames_sent.load(std::memory_order_relaxed) << ',';
    oss << "\"decode_queue_depth\":" << pipeline->decodedQueueDepth() << ',';
    oss << "\"decode_queue_limit\":" << pipeline->decodedQueueLimit() << ',';
    oss << "\"decode_drop_total\":" << pipeline->decodedDropTotal() << ',';
    oss << "\"last_infer_us\":" << metrics.last_inference_us.load(std::memory_order_relaxed) << ',';
    oss << "\"inference_fps\":" << pipeline->inferenceFps() << ',';
    oss << "\"last_total_us\":" << metrics.last_total_us.load(std::memory_order_relaxed);
    oss << '}';
}

} // namespace

// =============================================================================
// 主函数 — 程序入口
// =============================================================================
int main(int argc, char* argv[]) {
    // ---- 忽略 SIGPIPE：防止 RTSP/MQTT TCP 连接断开时进程被信号终止 ----
    std::signal(SIGPIPE, SIG_IGN);

    // ---- 阶段 1：初始化所有应用模块 ----
    // AppInitializer::initialize() 串联初始化：配置→存储→推理服务→视频管线→MQTT→流管理
    AppContext context;
    if (!AppInitializer::initialize(context)) {
        return 1; // 初始化失败，返回非零退出码
    }

    // ---- 阶段 2：按配置启动硬件看门狗 ----
    // 调试/试运行阶段默认关闭。否则应用崩溃会在几十秒后被放大成整机复位，
    // 丢失现场日志；量产确认稳定后再显式启用。
    const char* config_env = std::getenv("EDGE_SENSOR_CONFIG");
    const std::string config_path = config_env ? config_env : "../config/sensors.ini";
    IniConfig runtime_config;
    if (runtime_config.load(config_path) &&
        runtime_config.getBool("watchdog", "enable", false)) {
        const std::string device = runtime_config.getString(
            "watchdog", "device", "/dev/watchdog");
        openWatchdog(device.c_str(), runtime_config.getInt("watchdog", "timeout_sec", 30));
    } else {
        std::cout << "[Watchdog] disabled by configuration" << std::endl;
    }

    // ---- 阶段 3：通知 systemd 服务就绪 ----
    // Type=notify 模式要求服务主动上报 READY，否则 systemd 认为服务未启动
#if HAVE_SD_NOTIFY
    sd_notify(0, "READY=1");
    std::cout << "[Main] sd_notify(READY=1) sent" << std::endl;
#endif

    // ---- 阶段 4：注册信号处理器 ----
    std::signal(SIGINT, handleSignal);   // Ctrl+C / docker stop
    std::signal(SIGTERM, handleSignal);  // kill <pid> 默认信号

    // ---- 阶段 5：主循环 ----
    // 每 200ms 检查停止标志；每 5s（25 个周期）输出一次综合运行时统计
    int tick = 0;
    while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        tick++;

        if (tick >= 25) { // 25 × 200ms = 5s 统计周期
            tick = 0;
            kickWatchdog(); // 硬件看门狗心跳

            // systemd watchdog 心跳（与硬件 watchdog 同周期）
#if HAVE_SD_NOTIFY
            sd_notify(0, "WATCHDOG=1");
#endif

            // ---- 打印各管线指标摘要 ----
            if (context.external_rtsp_pipeline) {
                context.external_rtsp_pipeline->metrics().dump();
            }

            // ---- NPU 温度与节流状态 ----
            std::cerr << "[Thermal] NPU temp=" << context.thermal_manager.temperature()
                      << "°C level=" << static_cast<int>(context.thermal_manager.level())
                      << " throttle=" << context.thermal_manager.throttleFactor()
                      << std::endl;

            // ---- 采集各子系统运行时快照 ----
            const auto storage_stats = context.sqlite_store.runtimeStats();
            const auto mqtt_stats = context.mqtt_uploader.statsSnapshot();
            const int pending_upload = context.sqlite_store.pendingUploadCount();
            const uint64_t external_frames = context.external_rtsp_pipeline
                ? context.external_rtsp_pipeline->metrics().frames_in.load(std::memory_order_relaxed) : 0;
            const bool imx_online = context.imx415_pipeline && context.imx415_pipeline->online();
            const uint64_t imx_frames = context.imx415_pipeline
                ? context.imx415_pipeline->capturedFrames() : 0;
            const auto model_stats = context.inference_service
                ? context.inference_service->statsSnapshot()
                : pipeline::InferenceService::ModelRuntimeStats{};

            // 采集板端综合状态（CPU / 内存 / 磁盘 / 网络）
            const auto board_status = context.board_status_collector.sample(
                external_frames, imx_online, imx_frames, model_stats.count,
                model_stats.last_us, model_stats.average_us, model_stats.max_us);

            // ---- 板端状态写入 SQLite 并通知 MQTT 上传器 ----
            if (context.storage_enabled) {
                context.sqlite_store.insertRawRecord(
                    "device_status", board_status.timestamp_ms, board_status.toJson());
                if (context.mqtt_enabled) context.mqtt_uploader.notifyNewData();
            }

            // ---- 构建运行时统计 JSON（所有子系统指标拼装为一个 JSON 对象） ----
            std::ostringstream stats_json;
            stats_json << "{\"type\":\"runtime_stats\",\"ts_ms\":"
                       << toUnixMs(std::chrono::system_clock::now()) << ',';

            // 温控段
            stats_json << "\"thermal\":{";
            stats_json << "\"temp\":" << context.thermal_manager.temperature() << ',';
            stats_json << "\"level\":" << static_cast<int>(context.thermal_manager.level()) << ',';
            stats_json << "\"throttle\":" << context.thermal_manager.throttleFactor();
            stats_json << "},";

            // 板端综合状态
            stats_json << "\"board\":" << board_status.toJson() << ',';

            // 存储统计
            stats_json << "\"storage\":{";
            stats_json << "\"pending_upload\":" << pending_upload << ',';
            stats_json << "\"last_flush_us\":" << storage_stats.last_flush_us << ',';
            stats_json << "\"flush_count\":" << storage_stats.flush_count << ',';
            stats_json << "\"mark_count\":" << storage_stats.mark_count;
            stats_json << "},";

            // MQTT 上报统计
            stats_json << "\"mqtt\":{";
            stats_json << "\"reconnect_attempts\":" << mqtt_stats.reconnect_attempts << ',';
            stats_json << "\"publish_success\":" << mqtt_stats.publish_success << ',';
            stats_json << "\"publish_failed\":" << mqtt_stats.publish_failed << ',';
            stats_json << "\"stream_info_success\":" << mqtt_stats.stream_info_success << ',';
            stats_json << "\"stream_info_failed\":" << mqtt_stats.stream_info_failed;
            stats_json << "},";

            // 推理分发器统计
            stats_json << "\"inference_dispatch\":{";
            if (context.inference_dispatcher) {
                stats_json << "\"dropped\":" << context.inference_dispatcher->droppedCount() << ',';
                stats_json << "\"sample_ms\":" << context.inference_dispatcher->currentSampleMs();
            } else {
                stats_json << "\"dropped\":0,\"sample_ms\":0";
            }
            stats_json << "},";

            // 各视频管线实时指标
            stats_json << "\"pipelines\":{";
            bool first_pipeline = true;
            appendRtspStats(stats_json, "external_rtsp", context.external_rtsp_pipeline.get(), first_pipeline);

            if (context.imx415_pipeline) {
                if (!first_pipeline) stats_json << ',';
                first_pipeline = false;
                stats_json << "\"imx415\":{";
                stats_json << "\"capture_frames\":" << context.imx415_pipeline->capturedFrames() << ',';
                stats_json << "\"dropped_frames\":" << context.imx415_pipeline->droppedFrames() << ',';
                stats_json << "\"processed_fps\":" << context.imx415_pipeline->processedFps() << ',';
                stats_json << "\"inference_fps\":" << context.imx415_pipeline->inferenceFps();
                stats_json << '}';
            }

            if (context.mosaic_pipeline) {
                if (!first_pipeline) stats_json << ',';
                stats_json << "\"mosaic\":{";
                stats_json << "\"output_fps\":" << context.mosaic_pipeline->outputFps() << ',';
                stats_json << "\"encoded_fps\":" << context.mosaic_pipeline->encodedFps() << ',';
                stats_json << "\"sent_fps\":" << context.mosaic_pipeline->sentFps() << ',';
                stats_json << "\"frame_age_ms\":" << context.mosaic_pipeline->lastFrameAgeMs() << ',';
                stats_json << "\"frame_drops\":" << context.mosaic_pipeline->droppedFrames() << ',';
                stats_json << "\"rtsp_packet_drops\":" << context.mosaic_pipeline->droppedRtspPackets();
                stats_json << '}';
            }
            stats_json << "}}";

            std::cout << "[RuntimeStats] " << stats_json.str() << std::endl;
        }
    }

    // ---- 阶段 6：优雅退出 ----
    // 先关闭硬件看门狗（写魔术字符 'V'），再逆序释放各子系统
    closeWatchdog();
    AppInitializer::shutdown(context);

    return 0;
}
