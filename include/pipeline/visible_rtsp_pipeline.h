#pragma once  // 头文件保护宏，防止重复包含

// 标准库头文件
#include <atomic>               // std::atomic 原子操作类型
#include <array>                // std::array 固定大小数组
#include <chrono>               // std::chrono 时间工具
#include <cstdint>              // int64_t / uint8_t 等定宽整数
#include <functional>           // std::function 可调用对象包装器
#include <memory>               // std::unique_ptr 独占所有权智能指针
#include <mutex>                // std::mutex 互斥锁
#include <string>               // std::string 字符串类型
#include <thread>               // std::thread 线程管理
#include <condition_variable>   // std::condition_variable 条件变量

// OpenCV 头文件
#include <opencv2/core/mat.hpp>               // cv::Mat 图像矩阵

// 项目内部头文件
#include "config/ini_config.h"                // INI 配置文件解析工具
#include "data_processing/postprocess.h"      // detect_result_group_t 检测结果结构
#include "data_collection/stream_loader.h"    // StreamLoader RTSP 流加载器
#include "pipeline/pipeline_metrics.h"        // PipelineMetrics 管线指标 & RateCounter
#include "pipeline/frame_hub.h"               // FrameHub 多路帧数据交换中心
#include "storage/sqlite_store.h"             // 数据持久化存储

namespace pipeline { class InferenceService; }  // 前向声明推理服务类

namespace pipeline {  // 管线命名空间

/**
 * @struct VisibleInferenceConfig
 * @brief 可见光推理配置
 *
 * 控制推理的启停和帧间隔。
 */
struct VisibleInferenceConfig {
    bool enable = true;        // 是否启用推理
    int interval_ms = 0;       // 推理间隔（毫秒），0 表示每帧都推理
};

/**
 * @struct VisibleRtspPipelineConfig
 * @brief 可见光 RTSP 管线配置
 *
 * 封装可见光 RTSP 输入源的所有配置参数：
 * - RTSP 拉流地址和传输参数
 * - StreamLoader 缓冲区和超时参数
 * - 推理间隔配置
 */
struct VisibleRtspPipelineConfig {
    bool enable = false;                     // 是否启用此管线
    std::string url;                         // RTSP 拉流 URL 地址
    int pull_interval_ms = 10;               // 拉流间隔（毫秒），控制拉流帧率
    int loader_max_ready_depth = 4;          // StreamLoader 最大就绪帧深度
    int loader_rtbufsize_bytes = 1048576;    // StreamLoader 环形缓冲区大小（字节，默认 1MB）
    int loader_stimeout_us = 2000000;        // StreamLoader 单帧超时（微秒，默认 2s）
    int loader_max_delay_us = 100000;        // StreamLoader 最大允许延迟（微秒，默认 100ms）
    std::string loader_rtsp_transport = "tcp";  // RTSP 传输协议（tcp/udp）

    VisibleInferenceConfig inference;         // 推理配置（启用和间隔）
};

/**
 * @class VisibleRtspPipeline
 * @brief 可见光 RTSP 管线
 *
 * 负责从 RTSP URL 拉取视频流，解码后经 RGA 预处理和推理，
 * 将结果写入 FrameHub 供其他模块使用。
 *
 * 线程模型：
 *   1. loader 线程：StreamLoader 内部拉流解码
 *   2. 处理线程：读取解码帧 → RGA 转换 → FrameHub + 回调
 *   3. 推理线程：按间隔读取帧 → InferenceService → 缓存结果
 */
class VisibleRtspPipeline {
public:
    /**
     * @brief 构造函数
     * @param config 可见光 RTSP 管线配置
     */
    explicit VisibleRtspPipeline(VisibleRtspPipelineConfig config);

    /// 析构函数：确保 stop() 被调用
    ~VisibleRtspPipeline();

    /**
     * @brief 启动管线
     * @return true 启动成功；false RTSP 连接或初始化解码失败
     */
    bool start();

    /// 停止管线：断开 RTSP、等待线程退出、释放资源
    void stop();

    /// 解码帧回调函数类型：接收 BGR 帧、时间戳、monotonic 时间和覆盖层元数据
    using DecodedFrameCallback = std::function<void(const FrameHub::DmaFrame&,
        std::chrono::system_clock::time_point, int64_t, const FrameHub::FrameOverlay&)>;
    /// 设置解码帧回调函数（处理后的 BGR 帧由此输出）
    void setFrameCallback(DecodedFrameCallback cb);

    /// 推理统计回调函数类型
    using InferenceCallback = std::function<void(const data_lifecycle::InferenceStats&)>;
    /// 设置推理统计回调函数
    void setInferenceCallback(InferenceCallback cb);

    /// 设置流 ID（用于多路 RTSP 区分标识）
    void setStreamId(int id);

    /// 注入全局共享推理服务（外部持有生命周期）
    void setInferenceService(InferenceService* svc);

    /// 获取管线性能指标（只读引用）
    const PipelineMetrics& metrics() const { return metrics_; }

    /// 解码队列当前深度（待处理的解码帧数）
    std::uint32_t decodedQueueDepth() const;

    /// 解码队列容量上限
    std::uint32_t decodedQueueLimit() const;

    /// 解码队列累计丢弃帧数
    std::uint64_t decodedDropTotal() const;

    /// 推理帧率（滑动窗口计算）
    double inferenceFps() const { return inference_rate_.rate(); }

    /**
     * @brief 从 INI 文件加载配置
     * @param path    INI 文件路径
     * @param out     输出配置对象
     * @param section INI 节名称（如 "visible_rtsp"）
     * @return true 加载成功
     */
    static bool loadFromIni(const std::string& path, VisibleRtspPipelineConfig& out,
                            const std::string& section);

private:
    /// 处理线程主循环：从 StreamLoader 取帧、RGA 转换、回调输出、写 FrameHub
    void processingLoop();

    /// 推理线程主循环：按 interval 取帧送推理服务、缓存检测结果
    void inferenceLoop();

    /**
     * @struct InferenceTask
     * @brief 待推理任务描述
     *
     * 处理线程将待推理帧封装为此结构体，通知推理线程处理。
     */
    struct InferenceTask {
        StreamLoader::DecodedHwFrameDesc desc{};         // 解码后的硬件帧描述符（含 DMA fd）
        std::chrono::system_clock::time_point timestamp{}; // 帧时间戳
        int64_t capture_mono_ns = 0;                     // 采集时刻 monotonic 时间戳（纳秒）
    };

    VisibleRtspPipelineConfig config_;                   // 管线配置副本
    std::atomic<bool> running_{false};                   // 运行状态标志（所有线程的退出信号）

    std::unique_ptr<StreamLoader> loader_;               // RTSP 流加载器（独占所有权）
    std::thread loader_thread_;                          // StreamLoader 内部线程（由 loader 管理）
    std::thread processing_thread_;                      // 处理线程句柄
    std::thread inference_thread_;                       // 推理线程句柄

    std::mutex inference_mutex_;                         // 推理任务互斥锁
    std::condition_variable inference_cv_;               // 推理任务条件变量
    InferenceTask pending_inference_;                    // 待推理任务数据
    bool inference_pending_ = false;                     // 是否有待推理任务标志

    std::mutex detection_mutex_;                         // 检测缓存互斥锁
    std::array<int64_t, kMaxInferenceModels> cached_detection_mono_ns_{}; // 各模型检测结果时间戳
    std::array<detect_result_group_t, kMaxInferenceModels> cached_detections_{}; // 缓存的检测结果
    bool detection_cache_valid_ = false;                 // 检测缓存是否有效
    RateCounter frame_rate_;                             // 处理帧率计数器
    RateCounter inference_rate_;                         // 推理帧率计数器

    InferenceService* inference_service_ = nullptr;      // 共享推理服务（非持有，外部管理生命周期）

    DecodedFrameCallback frame_callback_;                // 帧输出回调函数
    InferenceCallback inference_callback_;               // 推理统计回调函数
    int stream_id_ = 0;                                  // 流 ID 标识
    PipelineMetrics metrics_;                            // 管线性能指标收集器

    static constexpr std::int64_t kMaxReuseDetectionMs = 700; // 检测结果最大复用时间（毫秒），超时须重新推理
};

} // namespace pipeline
