#pragma once  // 头文件保护宏，防止重复包含

// 标准库头文件
#include <atomic>      // std::atomic 原子操作类型，线程安全的标志位和计数器
#include <chrono>      // std::chrono 时间工具
#include <cstdint>     // int64_t / uint64_t 等定宽整数
#include <functional>
#include <memory>      // std::unique_ptr 独占所有权智能指针
#include <string>      // std::string 字符串类型
#include <thread>      // std::thread 线程管理
#include <vector>      // std::vector 动态数组，用于编码头缓冲

// OpenCV 头文件
#include <opencv2/core/mat.hpp>               // cv::Mat 图像矩阵

// 项目内部头文件
#include "pipeline/bounded_queue.hpp"          // BoundedQueue 线程安全有界队列
#include "pipeline/pipeline_metrics.h"         // RateCounter 帧率计数器
#include "pipeline/rga_preprocessor.h"
#include "data_processing/mpp_encoder.h"       // MppH264Encoder Rockchip MPP 硬编码器
#include "storage/event_recorder.h"            // EventRecorder 事件录像管理器

namespace pipeline {  // 管线命名空间

/**
 * @struct EncodedMediaConfig
 * @brief 编码媒体服务配置
 *
 * 封装 H.264 编码和 RTSP 推流的所有参数。
 */
struct EncodedMediaConfig {
    int width = 1280;                // 编码输出宽度（像素）
    int height = 720;                // 编码输出高度（像素）
    int fps = 30;                    // 编码目标帧率
    int bitrate = 2000000;           // 编码目标码率（bps，默认 2Mbps）
    bool enable_rtsp = true;         // 是否启用 RTSP 推流
    std::string rtsp_url;            // RTSP 推流目标地址（如 rtsp://mediamtx:8554/stream）
    int frame_queue_size = 2;        // 待编码帧队列容量
    int rtsp_packet_queue_size = 6;  // 编码后 RTSP 包队列容量
    recording::RecorderConfig recorder;  // 事件录像配置
};

/**
 * @struct ComposedFrame
 * @brief 待编码的合成帧
 *
 * 由MosaicStreamPipeline提交DMA图层，编码线程直接合成到MPP NV12输入。
 */
struct ComposedFrame {
    std::vector<RgaDmaComposeTask> dma_layers; // 首选：直接合成到MPP NV12输入
    std::function<void(cv::Mat&)> draw_luma_overlay; // 在NV12 Y平面绘制OSD，不复制整帧
    int64_t capture_mono_ns = 0;  // 原始采集 monotonic 时间戳（纳秒）
    int64_t capture_real_ms = 0;  // 原始采集真实时间（毫秒，epoch）
    uint64_t frame_id = 0;        // 帧唯一标识 ID

    bool directDma() const { return !dma_layers.empty(); }
};

/**
 * @class EncodedMediaService
 * @brief 编码媒体服务
 *
 * 负责将合成帧经 MPP 硬件编码为 H.264 后通过 RTSP 推送，
 * 同时支持事件触发的录像功能。
 *
 * 线程模型：
 *   1. 编码线程：从 frame_queue_ 取帧 → MPP 编码 → 推入 rtsp_queue_
 *   2. RTSP 线程：从 rtsp_queue_ 取包 → 发送到 MediaMTX
 *
 * 使用 BoundedQueue 线程安全队列连接各阶段，避免锁竞争。
 */
class EncodedMediaService {
public:
    /**
     * @brief 构造函数
     * @param config 编码媒体服务配置
     */
    explicit EncodedMediaService(EncodedMediaConfig config);

    /// 析构函数：确保 stop() 被调用
    ~EncodedMediaService();

    /**
     * @brief 启动编码和 RTSP 推流线程
     * @return true 启动成功；false MPP 编码器初始化失败
     */
    bool start();

    /// 停止编码和推流线程，释放资源
    void stop();

    /**
     * @brief 提交合成帧到编码队列
     * @param frame 合成帧（包含图像和时间戳）
     *
     * 生产者调用，帧进入 frame_queue_ 等待编码。
     */
    void submitFrame(ComposedFrame frame);

    /// 触发录像事件（开始/停止/分段录像）
    void trigger(const recording::Event& event);

    /**
     * @brief 更新检测状态（用于录像触发逻辑）
     * @param type            事件类型（如火焰检测、人员检测）
     * @param detected        是否检测到目标
     * @param capture_mono_ns monotonic 时间戳（纳秒）
     * @param capture_real_ms 真实时间戳（毫秒）
     * @param camera_id       摄像头 ID
     */
    void updateDetection(recording::EventType type, bool detected,
                         int64_t capture_mono_ns, int64_t capture_real_ms, int camera_id);

    /**
     * @brief 设置录像完成回调
     * @param callback 录像完成时调用的回调函数
     */
    void setRecordingCompletionCallback(recording::EventRecorder::CompletionCallback callback);

    /// 服务是否正在运行
    bool active() const { return running_.load(std::memory_order_acquire); }

    /// 编码帧队列累计丢弃数
    uint64_t droppedFrames() const {
        return frame_queue_.dropCount() + processing_drops_.load(std::memory_order_relaxed);
    }

    /// RTSP 包队列累计丢弃数
    uint64_t droppedRtspPackets() const { return rtsp_queue_.dropCount(); }

    /// 编码帧率（滑动窗口计算）
    double encodedFps() const { return encoded_rate_.rate(); }

    /// RTSP 推流帧率（滑动窗口计算）
    double sentFps() const { return sent_rate_.rate(); }

    /// 最后提交帧到当前时间的延迟（毫秒），用于判断管线是否卡顿
    int64_t lastFrameAgeMs() const { return last_frame_age_ms_.load(std::memory_order_relaxed); }

private:
    /// 编码线程主循环：取帧 → MPP 编码 → 生成 H.264 包 → 推入 rtsp_queue_
    void encoderLoop();

    /// RTSP 推流线程主循环：取包 → 发送到 RTSP 服务器
    void rtspLoop();

    EncodedMediaConfig config_;                                    // 编码服务配置副本
    BoundedQueue<ComposedFrame> frame_queue_;                      // 待编码帧队列（有界，drop-oldest 策略）
    BoundedQueue<std::shared_ptr<recording::EncodedPacket>> rtsp_queue_; // 编码后 RTSP 包队列
    recording::EventRecorder recorder_;                            // 事件录像管理器
    std::atomic<bool> running_{false};                             // 运行状态标志
    std::thread encoder_thread_;                                   // 编码线程句柄
    std::thread rtsp_thread_;                                      // RTSP 推流线程句柄
    std::unique_ptr<MppH264Encoder> encoder_;                      // MPP H.264 硬件编码器
    RgaPreprocessor direct_composer_;                              // DMA源直接合成到编码器NV12输入
    std::vector<uint8_t> codec_header_;                            // H.264 编码头（SPS/PPS），RTSP 初始化时发送
    RateCounter encoded_rate_;                                     // 编码帧率计数器
    RateCounter sent_rate_;                                        // 推流帧率计数器
    std::atomic<int64_t> last_frame_age_ms_{0};                   // 最后帧年龄（毫秒），原子操作，线程安全
    std::atomic<bool> force_next_idr_{false};                     // RTSP线程请求编码线程生成一次IDR
    std::atomic<uint64_t> processing_drops_{0};                  // RGA合成/OSD/编码失败丢帧
};

}  // namespace pipeline
