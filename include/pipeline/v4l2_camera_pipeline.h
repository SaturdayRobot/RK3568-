#pragma once  // 头文件保护宏，防止重复包含

// 标准库头文件
#include <atomic>               // std::atomic 原子操作类型，线程安全的计数器
#include <array>                // std::array 固定大小数组容器
#include <chrono>               // std::chrono 时间工具，用于时间戳和等待
#include <condition_variable>   // std::condition_variable 条件变量，线程间通知
#include <cstdint>              // int64_t 等定宽整数类型
#include <functional>           // std::function 可调用对象包装器
#include <memory>               // std::shared_ptr / std::unique_ptr 智能指针
#include <mutex>                // std::mutex 互斥锁
#include <string>               // std::string 字符串类型
#include <thread>               // std::thread 线程管理
#include <vector>               // std::vector 动态数组，用于缓冲区列表

// OpenCV 头文件
#include <opencv2/core/mat.hpp>               // cv::Mat 图像矩阵类型

// 项目内部头文件
#include "storage/sqlite_store.h"              // 数据持久化存储
#include "data_processing/postprocess.h"       // 后处理数据结构 detect_result_group_t
#include "pipeline/rga_preprocessor.h"         // RGA 硬件加速预处理模块
#include "pipeline/frame_hub.h"                // 多路帧数据交换中心
#include "pipeline/pipeline_metrics.h"         // 管线性能指标 (RateCounter, ScopedTimer)

namespace pipeline {  // 管线命名空间
class InferenceService;  // 前向声明推理服务类

/**
 * @struct V4l2CameraConfig
 * @brief V4L2 摄像头采集配置结构体
 *
 * 封装所有 V4L2 设备采集参数，
 * 可从 INI 配置文件加载。
 */
struct V4l2CameraConfig {
    bool enable = false;                     // 是否启用此摄像头管线
    std::string device = "/dev/video0";      // V4L2 设备路径
    int width = 1920;                        // 采集分辨率宽度（像素）
    int height = 1080;                       // 采集分辨率高度（像素）
    int fps = 30;                            // 采集帧率（FPS）
    int processing_width = 1280;             // RGA 中间 BGR 宽度；无需保留超过最终画布的像素
    int processing_height = 720;             // RGA 中间 BGR 高度
    std::string pixel_format = "NV12";       // 采集像素格式（NV12 为 YUV 4:2:0 半平面格式）
    std::string color_space = "auto";        // 颜色空间（auto 表示自动检测）
    int rotation = 0;                        // 顺时针旋转角度：0/90/180/270
    int inference_interval_ms = 200;         // 推理间隔（毫秒），控制推理帧率
};

/**
 * @class V4l2CameraPipeline
 * @brief V4L2 摄像头采集管线
 *
 * 负责通过 V4L2 接口从摄像头采集视频帧，经 RGA 预处理后
 * 送入推理服务和 FrameHub。内部使用三线程模型：
 *   1. 采集线程：从内核 V4L2 缓冲区取出原始帧
 *   2. 处理线程：RGA 转换 + 写入 FrameHub + 触发回调
 *   3. 推理线程：按间隔发送帧到 InferenceService 并缓存检测结果
 */
class V4l2CameraPipeline {
public:
    /// 帧输出回调函数类型：接收 BGR 帧、时间戳、monotonic 时间和覆盖层元数据
    using FrameCallback = std::function<void(const cv::Mat&,
        std::chrono::system_clock::time_point, int64_t, const FrameHub::FrameOverlay&)>;
    /// 推理统计回调函数类型：接收推理统计数据
    using InferenceCallback = std::function<void(const data_lifecycle::InferenceStats&)>;

    /**
     * @brief 构造函数
     * @param config V4L2 摄像头配置
     *
     * 初始化配置、RGA 预处理器实例和线程状态。
     */
    explicit V4l2CameraPipeline(V4l2CameraConfig config);

    /// 析构函数：确保 stop() 被调用，释放 V4L2 设备和线程资源
    ~V4l2CameraPipeline();

    /**
     * @brief 从 INI 文件加载配置
     * @param path INI 配置文件路径
     * @param out  输出配置对象
     * @return true 加载成功；false 文件不存在或解析失败
     */
    static bool loadFromIni(const std::string& path, V4l2CameraConfig& out);

    /// 注入共享推理服务（外部持有生命周期）
    void setInferenceService(InferenceService* service) { inference_service_ = service; }

    /// 设置帧回调函数（处理完的 BGR 帧将由此回调输出）
    void setFrameCallback(FrameCallback cb) { frame_callback_ = std::move(cb); }

    /// 设置推理统计回调函数
    void setInferenceCallback(InferenceCallback cb) { inference_callback_ = std::move(cb); }

    /**
     * @brief 启动管线
     * @return true 启动成功；false V4L2 设备打开失败
     *
     * 打开 V4L2 设备、初始化 RGA、启动三个工作线程。
     */
    bool start();

    /// 停止管线：关闭设备、等待线程退出、释放资源
    void stop();

    /// 管线是否在线（设备已打开且采集线程正常运行）
    bool online() const { return online_.load(std::memory_order_acquire); }

    /// 累计采集帧数
    uint64_t capturedFrames() const { return captured_frames_.load(); }

    /// 累计丢弃帧数
    uint64_t droppedFrames() const { return dropped_frames_.load(); }

    /// 处理帧率（滑动窗口计算）
    double processedFps() const { return frame_rate_.rate(); }

    /// 推理帧率（滑动窗口计算）
    double inferenceFps() const { return inference_rate_.rate(); }

private:
    /**
     * @struct Buffer
     * @brief V4L2 用户空间缓冲区描述
     *
     * 封装 mmap 映射后的缓冲区地址、长度和 DMA fd。
     */
    struct Buffer {
        void* address = nullptr;  // mmap 映射后的用户空间虚拟地址
        size_t length = 0;        // 缓冲区长度（字节）
        int dma_fd = -1;          // DMA-BUF 导出文件描述符（-1 表示未导出）
    };

    /// 打开 V4L2 设备，查询能力、设置格式、请求并映射缓冲区
    bool openDevice();

    /// 关闭 V4L2 设备，释放所有缓冲区
    void closeDevice();

    /// 将指定索引的缓冲区重新加入 V4L2 采集队列
    bool queueBuffer(uint32_t index);

    /// 采集线程主循环：从 V4L2 出队原始帧，通知处理线程
    void captureLoop();

    /// 处理线程主循环：RGA 转换、写入 FrameHub、触发回调
    void processingLoop();

    /// 推理线程主循环：按配置间隔将帧送入推理服务
    void inferenceLoop();

    /**
     * @brief 原始帧格式转换
     * @param data  原始帧数据指针
     * @param bytes 原始数据字节数
     * @param bgr   输出 BGR 图像
     * @return true 转换成功
     *
     * 使用 RGA 硬件将 NV12/其他格式转换为 BGR888。
     */
    bool convertFrame(const void* data, size_t bytes, cv::Mat& bgr) const;

    // ── 配置与设备 ──
    V4l2CameraConfig config_;          // 摄像头配置参数
    int fd_ = -1;                      // V4L2 设备文件描述符（-1 表示未打开）
    uint32_t buffer_type_ = 0;         // V4L2 缓冲区类型（V4L2_BUF_TYPE_VIDEO_CAPTURE[_MPLANE]）
    uint32_t num_planes_ = 1;          // 每帧平面数（单平面=1，多平面>1）
    uint32_t fourcc_ = 0;             // 像素格式四字符码（V4L2_PIX_FMT_xxx）
    int stride_ = 0;                   // 行步长（字节），用于计算每行数据偏移
    int height_stride_ = 0;            // 高度步长（像素），多平面格式的垂直对齐
    size_t size_image_ = 0;            // 单帧图像数据大小（字节）
    std::vector<Buffer> buffers_;      // V4L2 用户空间缓冲区列表

    // ── RGA 预处理器 ──
    RgaPreprocessor camera_rga_;       // RGA 硬件加速预处理器实例
    bool camera_rga_ready_ = false;    // RGA 预处理是否初始化就绪

    // ── 运行状态 ──
    std::atomic<bool> running_{false};       // 管线运行标志（所有线程的退出信号）
    std::atomic<bool> online_{false};         // 设备在线标志
    std::atomic<uint64_t> captured_frames_{0}; // 累计采集帧数（原子操作，线程安全）
    std::atomic<uint64_t> dropped_frames_{0};  // 累计丢弃帧数（原子操作）

    // ── 工作线程 ──
    std::thread capture_thread_;       // 采集线程句柄
    std::thread processing_thread_;    // 处理线程句柄
    std::thread inference_thread_;     // 推理线程句柄

    // ── 推理队列 ──
    std::mutex inference_mutex_;                          // 推理帧互斥锁
    std::condition_variable inference_cv_;                // 推理帧条件变量（通知有新帧待推理）
    cv::Mat inference_frame_;                             // 待推理的 BGR 帧副本
    std::chrono::system_clock::time_point inference_timestamp_{}; // 待推理帧的时间戳
    int64_t inference_mono_ns_ = 0;                       // 待推理帧的 monotonic 时间戳（纳秒）
    bool inference_pending_ = false;                      // 是否有待处理的推理任务

    // ── 采集-处理队列 ──
    mutable std::mutex latest_mutex_;                     // 最新采集帧互斥锁
    std::condition_variable latest_cv_;                   // 最新采集帧条件变量
    int latest_buffer_index_ = -1;                        // 最新采集缓冲区索引
    size_t latest_bytes_used_ = 0;                        // 最新采集帧实际数据字节数
    std::chrono::system_clock::time_point latest_timestamp_{}; // 最新采集帧时间戳
    int64_t latest_mono_ns_ = 0;                          // 最新采集帧 monotonic 时间戳（纳秒）
    uint64_t latest_seq_ = 0;                             // 最新采集帧序号

    // ── 检测结果缓存 ──
    bool detection_cache_valid_ = false;                  // 检测缓存是否有效
    std::mutex detection_mutex_;                          // 检测缓存互斥锁
    std::array<int64_t, kMaxInferenceModels> cached_detection_mono_ns_{}; // 各模型检测结果的时间戳
    std::array<detect_result_group_t, 4> cached_detections_{};           // 缓存的检测结果
    RateCounter frame_rate_;                              // 处理帧率计数器
    RateCounter inference_rate_;                          // 推理帧率计数器

    // ── 外部依赖 ──
    InferenceService* inference_service_ = nullptr;  // 共享推理服务指针（非持有，外部管理生命周期）
    FrameCallback frame_callback_;                    // 帧输出回调函数
    InferenceCallback inference_callback_;            // 推理统计回调函数
};
}  // namespace pipeline
