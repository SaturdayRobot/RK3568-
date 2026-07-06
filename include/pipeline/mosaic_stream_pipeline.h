#pragma once  // 头文件保护宏，防止重复包含

/**
 * @file mosaic_stream_pipeline.h
 * @brief 马赛克流处理管道
 *
 * 该文件定义了马赛克流处理管道类和相关配置结构，用于将多个视频流
 * 合成为一个马赛克画面，并通过RTSP推送到流媒体服务器。
 *
 * 主要功能：
 * 1. 多视频流合成（A路 RTSP + B路 IMX415 拼成 2x2 或左右布局）
 * 2. 马赛克画面生成（RGA 硬件加速 resize + blit）
 * 3. RTSP 推流（通过 EncodedMediaService 编码后推送）
 * 4. 帧率控制与性能监控
 */

// 标准库头文件
#include <array>       // std::array 固定大小数组
#include <atomic>      // std::atomic 原子操作类型
#include <cstdint>     // int64_t / uint64_t 等定宽整数
#include <memory>      // std::shared_ptr / std::unique_ptr 智能指针
#include <string>      // std::string 字符串类型
#include <thread>      // std::thread 工作线程管理
#include <tuple>       // std::tuple 用于批量 RGA 任务的数据组合
#include <vector>      // std::vector 动态数组

// OpenCV 头文件
#include <opencv2/core/mat.hpp>  // cv::Mat 图像矩阵 & cv::Rect 矩形区域

// 项目内部头文件
#include "config/ini_config.h"                // INI 配置文件解析
#include "pipeline/encoded_media_service.h"   // EncodedMediaService 编码推流服务
#include "pipeline/frame_hub.h"               // FrameHub 多路帧数据交换中心
#include "pipeline/pipeline_metrics.h"        // RateCounter 帧率计数器

namespace pipeline {  // 管线命名空间

/**
 * @struct MosaicStreamConfig
 * @brief 马赛克流处理管道配置结构体
 *
 * 存储马赛克流处理管道的配置参数，包括分辨率、帧率、码率、
 * RTSP推流地址和推流开关等。
 */
struct MosaicStreamConfig {
    bool enable = false;                    // 是否启用马赛克流处理管道
    int width = 1280;                       // 输出分辨率宽度（像素）
    int height = 720;                       // 输出分辨率高度（像素）
    int fps = 15;                           // 输出帧率（FPS）
    int bitrate = 2000000;                  // 输出码率（bps，默认 2Mbps）
    std::string rtsp_url;                   // RTSP 推流地址（如 rtsp://mediamtx:8554/mosaic）
    bool enable_rtsp = true;                // 是否启用 RTSP 推流
    std::string input_mode = "side_by_side"; // 输入布局模式：side_by_side（左右）/ vertical（上下）
    bool preserve_aspect_ratio = true;       // 是否等比缩放并居中补边（禁止画面拉伸变形）
    std::string aspect_mode = "contain";     // 宽高比适配模式：cover（满幅裁剪）/ contain（完整补边）
    bool draw_overlay = true;                // 是否在最终画布上绘制检测框与性能指标叠加层
    int stream_queue_size = 3;              // 推流队列深度（待编码帧缓冲数）
    int rtsp_packet_queue_size = 6;          // 发往 MediaMTX 的编码包低延迟队列深度
    bool stream_drop_oldest_when_full = true; // 队列满时是否丢弃旧帧（true=丢弃旧帧保留新帧）
    bool stream_consume_latest_only = true;   // 消费端是否每轮只处理最新帧（跳过积压的旧帧）
    bool stream_resize_on_mismatch = false;  // 分辨率不匹配时是否自动 resize 适配
    recording::RecorderConfig recorder;       // 事件录像配置
    bool sync_enabled = true;                 // 是否启用多路帧时间同步
    int sync_threshold_ms = 15;              // 帧同步的最大允许时间差（毫秒）
    int sync_queue_depth = 4;                // 帧同步历史队列深度（保留最近 N 帧用于对齐）
};

/**
 * @class MosaicStreamPipeline
 * @brief 马赛克流处理管道类
 *
 * 负责将多个视频流（可见光 RTSP + IMX415 摄像头）合成为一个马赛克画面，
 * 并通过 RTSP 推送到流媒体服务器。
 *
 * 工作流程：
 *   1. 从 FrameHub 获取多路帧快照（支持时间同步）
 *   2. 通过 RGA 硬件或 CPU 回退将各帧缩放拼接到画布 ROI
 *   3. 在画布上绘制检测框叠加层和性能指标
 *   4. 将合成帧提交给 EncodedMediaService 进行 H.264 编码和 RTSP 推流
 *
 * 支持自定义分辨率、帧率和码率，支持动态开关推流。
 */
class MosaicStreamPipeline {
public:
    /**
     * @brief 构造函数
     * @param config 马赛克流处理管道配置
     * @param hub    FrameHub 共享指针（提供多路帧源数据）
     */
    MosaicStreamPipeline(MosaicStreamConfig config, std::shared_ptr<FrameHub> hub);

    /// 析构函数：确保 stop() 被调用，释放线程和 EncodedMediaService
    ~MosaicStreamPipeline();

    /**
     * @brief 启动马赛克流处理管道
     * @return true 启动成功；false EncodedMediaService 初始化或线程创建失败
     */
    bool start();

    /// 停止马赛克流处理管道：等待工作线程退出、停止编码推流服务
    void stop();

    /**
     * @brief 启用或禁用流媒体推送
     * @param enable true 启用推流；false 禁用推流
     *
     * 动态控制 RTSP 推流的开启和关闭。
     * 启用时会初始化推流管理器并开始推流；
     * 禁用时会停止推流并释放相关资源。
     */
    void setStreamingEnabled(bool enable);

    /// 触发录像事件（开始/停止/分段录像）
    void triggerRecording(const recording::Event& event);

    /**
     * @brief 更新检测状态（用于录像触发逻辑）
     * @param type            事件类型
     * @param detected        是否检测到目标
     * @param capture_mono_ns monotonic 时间戳（纳秒）
     * @param capture_real_ms 真实时间戳（毫秒）
     * @param camera_id       摄像头 ID
     */
    void updateDetection(recording::EventType type, bool detected,
                         int64_t capture_mono_ns, int64_t capture_real_ms, int camera_id);

    /// 设置录像完成回调
    void setRecordingCompletionCallback(recording::EventRecorder::CompletionCallback callback);

    /// 输出帧率（滑动窗口计算）
    double outputFps() const { return output_rate_.rate(); }

    /// 编码帧队列累计丢弃数
    uint64_t droppedFrames() const {
        return media_service_ ? media_service_->droppedFrames() : 0;
    }

    /// RTSP 包队列累计丢弃数
    uint64_t droppedRtspPackets() const {
        return media_service_ ? media_service_->droppedRtspPackets() : 0;
    }

    /// 编码帧率
    double encodedFps() const { return media_service_ ? media_service_->encodedFps() : 0.0; }

    /// 推流帧率
    double sentFps() const { return media_service_ ? media_service_->sentFps() : 0.0; }

    /// 最后帧年龄（毫秒），用于判断管线是否卡顿
    int64_t lastFrameAgeMs() const {
        return media_service_ ? media_service_->lastFrameAgeMs() : 0;
    }

    /**
     * @brief 从 INI 文件加载配置
     * @param path INI 配置文件路径
     * @param out  输出配置对象
     * @return true 加载成功；false 文件不存在或解析失败
     *
     * 读取配置文件中的 [mosaic_stream] 节，解析相关参数。
     */
    static bool loadFromIni(const std::string& path, MosaicStreamConfig& out);

private:
    /**
     * @brief 管道处理主循环
     *
     * 在独立线程中运行，负责：
     * 1. 按照配置的帧率控制循环周期
     * 2. 调用 composeFrame() 合成马赛克画面
     * 3. 将合成画面提交给独立编码服务，编码包再分发给 RTSP 与录像线程
     */
    void loop();

    /**
     * @brief 合成马赛克帧
     * @return 合成后的马赛克帧（cv::Mat BGR 图像）
     *
     * 从 FrameHub 获取可见光与多路 RTSP 视频流，
     * 将它们缩放并拼接成指定布局（side_by_side / vertical）。
     * 如果某路视频流获取失败，对应区域将显示黑色。
     */
    cv::Mat composeFrame();

    /// RGA 将源 ROI 直接缩放写入完整画布 ROI，避免 tile 中间缓冲再 copyTo（单图拼接）
    bool rgaBlit(const cv::Mat& src, cv::Mat& canvas,
                 const cv::Rect& source_roi, const cv::Rect& destination_roi);

    /// RGA 批量处理：resize + 直接拼接（一次性提交所有任务，减少 RGA 调用次数）
    bool rgaBatchCompose(const std::vector<std::tuple<cv::Mat, int, int, cv::Rect>>& tasks,
                         cv::Mat& canvas);

    /// 探测 RGA 硬件可用性（仅调用一次，结果缓存到 rga_available_）
    void probeRga();

    /// 更新遥测数据（CPU/NPU/RGA/DDR 占用率、丢帧率等）
    void updateTelemetry(const std::vector<FrameHub::FrameSnapshot>& snapshot);

    /// 在画布上绘制全局叠加层（检测框、FPS 等性能指标文字）
    void drawGlobalOverlay(cv::Mat& canvas) const;

    // ── 配置与依赖 ──
    MosaicStreamConfig config_;              // 配置参数副本
    std::shared_ptr<FrameHub> hub_;         // 帧中心共享指针（外部共享，不拥有所有权）

    // ── 运行状态 ──
    std::atomic<bool> running_{false};       // 运行状态标志（工作线程的退出信号）
    std::thread worker_;                     // 工作线程句柄
    std::unique_ptr<EncodedMediaService> media_service_; // 编码推流服务（独占所有权）

    // ── 录像 ──
    recording::EventRecorder::CompletionCallback recording_completion_callback_; // 录像完成回调
    uint64_t composed_frame_id_ = 0;          // 合成帧 ID 计数器（单调递增）

    // ── 画布缓冲 ──
    // 预分配三缓冲，避免逐帧分配；compose 不再整帧复制到下一画布
    cv::Mat mosaic_canvas_[3];               // 三缓冲画布数组（生产者-消费者解耦）
    std::atomic<int> write_idx_{0};           // 当前写入画布索引（0/1/2 轮转）
    std::array<cv::Mat, 2> tile_bufs_;        // 双路预分配 tile 临时缓冲（避免重复分配）

    // ── RGA 状态 ──
    bool rga_available_ = false;              // RGA 硬件是否可用（探测结果）
    bool rga_probed_ = false;                 // 是否已完成 RGA 硬件探测

    // ── 帧同步统计 ──
    uint64_t sync_pairs_ = 0;                 // 成功同步的帧对总数
    uint64_t sync_drop_first_ = 0;            // 第一路因同步丢弃的帧数
    uint64_t sync_drop_second_ = 0;           // 第二路因同步丢弃的帧数
    int64_t sync_delta_total_ns_ = 0;         // 同步时间差累计（纳秒），用于计算平均差
    int64_t sync_delta_max_ns_ = 0;           // 同步时间差最大值（纳秒）
    int sync_fail_streak_ = 0;                // 连续同步失败次数（用于降级决策）
    int sync_skip_cooldown_ = 0;              // 跳过同步剩余帧数（避免超时拖累帧率）

    // ── 帧时间戳 ──
    int64_t last_composed_mono_ns_ = 0;       // 上次合成帧的 monotonic 时间戳（纳秒）
    int64_t last_composed_real_ms_ = 0;       // 上次合成帧的真实时间戳（毫秒）
    RateCounter output_rate_;                 // 输出帧率计数器

    // ── 遥测数据 ──
    struct CanvasTelemetry {
        double cpu_percent = 0.0;              // CPU 使用率百分比
        double npu_percent = 0.0;              // NPU 使用率百分比
        double rga_percent = 0.0;              // RGA 使用率百分比
        double ddr_percent = 0.0;              // DDR 带宽使用率百分比
        double drop_percent = 0.0;             // 帧丢弃率百分比
        uint64_t previous_captured = 0;        // 上次采样的累计采集帧数（用于计算增量）
        uint64_t previous_dropped = 0;         // 上次采样的累计丢弃帧数（用于计算增量）
        uint64_t previous_cpu_total = 0;       // 上次采样的 CPU 总时间（用于计算占用率）
        uint64_t previous_cpu_idle = 0;        // 上次采样的 CPU 空闲时间（用于计算占用率）
        std::chrono::steady_clock::time_point sampled_at{}; // 上次采样时间点
    } telemetry_;
};

} // namespace pipeline
