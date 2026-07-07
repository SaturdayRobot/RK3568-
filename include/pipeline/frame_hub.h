#pragma once  // 头文件保护宏，防止重复包含

// 标准库头文件
#include <array>                // std::array 固定大小数组容器
#include <atomic>               // std::atomic 原子操作类型，线程安全
#include <chrono>               // std::chrono 时间相关工具
#include <condition_variable>   // std::condition_variable 条件变量，线程间通知
#include <cstdint>              // int64_t, uint64_t 等定宽整数类型
#include <deque>                // std::deque 双端队列容器
#include <memory>               // std::shared_ptr 共享指针
#include <mutex>                // std::mutex 互斥锁
#include <vector>               // std::vector 动态数组容器

// 项目内部头文件
#include <opencv2/core/mat.hpp>               // OpenCV 图像矩阵 cv::Mat
#include "data_processing/postprocess.h"       // 后处理数据结构 detect_result_group_t
#include "pipeline/inference_service.h"        // 推理服务依赖 (kMaxInferenceModels)

namespace pipeline {  // 管线命名空间

/**
 * @enum FrameSource
 * @brief 帧源枚举类型
 *
 * 标识每帧数据来自哪个摄像头/输入源，
 * 对应 FrameHub 内部 slots_ 数组的下标索引。
 */
enum class FrameSource {
    ExternalRtsp = 0,  // 外部 RTSP 输入源 (A路)
    Imx415 = 1,        // IMX415 内置摄像头 (B路)
    Count = 2          // 帧源总数，用于定义数组长度
};

enum class DmaPixelFormat : uint8_t {
    NV12 = 0,
    NV21 = 1,
    NV16 = 2,
};

enum class DmaColorSpace : uint8_t {
    Bt601Limited = 0,
    Bt601Full = 1,
    Bt709Limited = 2,
};

/**
 * @class FrameHub
 * @brief 多路帧数据交换中心（帧枢纽）
 *
 * 充当各管线模块之间的帧数据共享总线：
 * - 采集线程通过 update() 写入最新帧
 * - 消费者线程通过 get()/snapshot() 等接口读取帧
 * - 每个帧源有独立槽位 Slot，避免读写竞争
 * - 支持帧同步、覆盖层信息传递等功能
 */
class FrameHub {
public:
    struct DmaFrame {
        int fd = -1;
        int width = 0;
        int height = 0;
        int width_stride = 0;
        int height_stride = 0;
        size_t buffer_size = 0;
        int rotation = 0;  // 顺时针0/90/180/270；RGA合成时一次完成
        DmaPixelFormat format = DmaPixelFormat::NV12;
        DmaColorSpace color_space = DmaColorSpace::Bt709Limited;
        std::shared_ptr<void> lease; // 持有期间采集端不得覆盖/回收DMA缓冲

        bool valid() const {
            return fd >= 0 && width > 0 && height > 0 &&
                   width_stride >= width && height_stride >= height &&
                   buffer_size > 0 && lease != nullptr;
        }
        int logicalWidth() const { return rotation == 90 || rotation == 270 ? height : width; }
        int logicalHeight() const { return rotation == 90 || rotation == 270 ? width : height; }
    };

    /**
     * @struct FrameOverlay
     * @brief 帧覆盖层元数据
     *
     * 携带与帧图像关联的非像素信息：
     * 检测结果、FPS 指标、帧计数等。
     * 随帧一起写入 FrameHub，消费者无需单独查询。
     */
    struct FrameOverlay {
        std::array<detect_result_group_t, kMaxInferenceModels> detections{};  // 各模型的检测结果数组
        bool detections_valid = false;     // 检测结果是否有效标志
        int source_width = 0;             // 原始帧宽度（像素）
        int source_height = 0;            // 原始帧高度（像素）
        double frame_fps = 0.0;           // 当前帧源采集帧率
        double inference_fps = 0.0;       // 当前推理帧率
        int64_t detection_mono_ns = 0;    // 检测结果对应的 monotonic 时间戳（纳秒）
        uint64_t frames_captured = 0;     // 累计采集帧数
        uint64_t frames_dropped = 0;      // 累计丢弃帧数
    };

    /**
     * @struct FrameSnapshot
     * @brief 帧快照
     *
     * 一次调用 snapshot() 返回的结构体，
     * 将帧指针、时间戳、序号和覆盖层绑定在一起。
     */
    struct FrameSnapshot {
        std::shared_ptr<cv::Mat> frame;                          // 帧图像共享指针（避免拷贝）
        DmaFrame dma;                                            // 首选DMA分发帧；frame为兼容回退
        std::chrono::system_clock::time_point timestamp{};       // 系统时钟时间戳
        int64_t capture_mono_ns = 0;                             // 采集时的 monotonic 时间戳（纳秒）
        uint64_t seq = 0;                                        // 帧序号（单调递增）
        FrameOverlay overlay;                                    // 帧覆盖层元数据
    };

    /**
     * @struct Slot
     * @brief 帧槽位结构体（per-slot 独立锁）
     *
     * 每个帧源对应一个 Slot，拥有独立的互斥锁和条件变量，
     * 读写不同帧源之间互不阻塞。同时维护 sync_queue 用于帧同步。
     */
    struct Slot {
        mutable std::mutex              mtx;           // 该槽位独立锁，保护本槽位所有字段
        mutable std::condition_variable cv;            // 新帧通知，消费者可 wait 等待
        std::shared_ptr<cv::Mat>        frame;         // 帧数据（shared_ptr 避免重复拷贝）
        DmaFrame                        dma;           // DMA帧及采集缓冲生命周期令牌
        std::chrono::system_clock::time_point timestamp{}; // 帧时间戳（系统时钟）
        int64_t capture_mono_ns = 0;                   // 采集时 monotonic 时间戳（纳秒），用于同步计算
        uint64_t                        seq = 0;       // 帧序号（单调递增），消费者通过序号判断是否有新帧
        FrameOverlay                    overlay;       // 帧覆盖层元数据
        bool                            has_frame = false;  // 是否有有效帧数据
        std::deque<FrameSnapshot>       sync_queue;    // 同步用帧历史队列，保留最近 N 帧用于时间对齐
        uint64_t                        sync_overflow_drops = 0;  // 同步队列溢出丢弃计数
    };

    /**
     * @brief 更新帧数据（move 语义，仅保留该入口）
     * @param source 帧源标识
     * @param frame 帧图像（move 接管所有权，避免像素深拷贝）
     * @param timestamp 系统时钟时间戳
     * @param capture_mono_ns 采集时刻 monotonic 时间戳（纳秒）
     * @param overlay 帧覆盖层元数据
     *
     * 写入后唤醒所有正在等待该帧源的消费者。
     */
    void update(FrameSource source, cv::Mat&& frame,
                std::chrono::system_clock::time_point timestamp,
                int64_t capture_mono_ns,
                FrameOverlay overlay);

    void updateDma(FrameSource source, DmaFrame frame,
                   std::chrono::system_clock::time_point timestamp,
                   int64_t capture_mono_ns,
                   FrameOverlay overlay);

    /// 释放槽位和同步队列持有的全部帧租约，停机时必须先于采集设备关闭调用。
    void clear();

    /**
     * @brief 获取帧数据（非阻塞，shared_ptr 版本）
     * @param source 帧源标识
     * @param frame_out 输出帧图像（shared_ptr，不做 clone）
     * @param ts_out 输出时间戳
     * @return true 成功获取帧；false 尚无帧数据
     *
     * 返回的 shared_ptr 与 Slot 内部共享所有权，读取期间不会被覆盖。
     */
    bool get(FrameSource source, std::shared_ptr<cv::Mat>& frame_out,
             std::chrono::system_clock::time_point& ts_out) const;

    /**
     * @brief 获取帧数据（兼容原接口：返回 cv::Mat 浅拷贝）
     * @param source 帧源标识
     * @param frame_out 输出帧图像（浅拷贝，Mat header 复制但像素共享）
     * @param ts_out 输出时间戳
     * @return true 成功获取帧；false 尚无帧数据
     *
     * @note 调用方若需要修改帧数据，应自行 clone()
     */
    bool get(FrameSource source, cv::Mat& frame_out,
             std::chrono::system_clock::time_point& ts_out) const;

    /**
     * @brief 等待指定帧源的新帧到达（阻塞）
     * @param source 帧源标识
     * @param last_seq 上次获取的帧序号（传入 0 表示获取任何已有帧）
     * @param timeout 最大等待时间（毫秒）
     * @return true 有新帧到达；false 超时
     *
     * 消费者先调用此方法等待新帧，再调用 get() 读取帧数据。
     */
    bool waitForNew(FrameSource source, uint64_t last_seq,
                    std::chrono::milliseconds timeout = std::chrono::milliseconds(200)) const;

    /**
     * @brief 一次性读取多路最新帧快照
     * @param sources 需要读取的源列表
     * @param out 对应源的快照输出（与 sources 顺序一致）
     *
     * 在单次加锁范围内读取所有指定帧源的最新帧，
     * 保证获得的多路数据来自同一时刻附近。
     */
    void snapshot(const std::vector<FrameSource>& sources,
                  std::vector<FrameSnapshot>& out) const;

    /**
     * @struct SyncResult
     * @brief 帧同步结果
     *
     * 包含两路对齐后的帧快照、时间差和丢弃统计，
     * 用于 mosaic 合成等需要多路时间对齐的场景。
     */
    struct SyncResult {
        FrameSnapshot first;       // 第一路对齐帧快照
        FrameSnapshot second;      // 第二路对齐帧快照
        int64_t delta_ns = 0;      // 两路帧之间的时间差（纳秒）
        uint64_t dropped_first = 0;  // 第一路因同步丢弃的帧数
        uint64_t dropped_second = 0; // 第二路因同步丢弃的帧数
    };

    /**
     * @brief 尝试获取时间对齐的两路帧
     * @param first  第一帧源
     * @param second 第二帧源
     * @param threshold_ns 允许的最大时间差（纳秒）
     * @param out    同步结果输出
     * @return true 成功获取同步帧对；false 当前无可对齐的帧对
     *
     * 从两路的 sync_queue 中查找时间戳差距在阈值内的帧对。
     */
    bool takeSynchronized(FrameSource first, FrameSource second,
                          int64_t threshold_ns, SyncResult& out);

    /// 设置同步队列深度（最少为 2）
    void setSyncQueueDepth(size_t depth) { sync_queue_depth_ = depth > 1 ? depth : 2; }

    /// 获取指定帧源当前帧序号（用于消费者判断是否有新帧）
    uint64_t seq(FrameSource source) const;

private:
    std::array<Slot, static_cast<size_t>(FrameSource::Count)> slots_{};  // 帧源槽位数组，每个帧源一个 Slot
    std::atomic<size_t> sync_queue_depth_{4};                            // 同步队列深度，控制保留历史帧数
};

} // namespace pipeline
