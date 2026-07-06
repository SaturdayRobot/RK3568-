/**
 * @file stream_loader.h
 * @brief 视频流加载与管理模块
 *
 * 该模块负责单路视频流的打开、读取与MPP硬件解码，是整个系统的数据源入口。
 * 使用FFmpeg进行流读取，MPP（Rockchip平台）进行硬件解码。
 *
 * 典型使用流程：构造 -> open() -> operator()（线程循环） -> close() -> 析构
 *
 * 核心设计要点：
 * - 零拷贝DMA-BUF传递：帧数据以文件描述符形式传递，避免GPU/CPU间拷贝
 * - 生产者-消费者模式：解码回调（生产者）写入ready_queue_，
 *   上层管线通过waitAndGetDecodedFrame（消费者）阻塞等待取出帧
 * - 帧描述符回收机制：消费者处理后调用releaseDecodedFrame归还槽位
 * - 自动重连：operator()线程检测断流后使用指数退避策略自动恢复
 */

#pragma once

// FFmpeg 相关头文件（C库，需要 extern "C" 包裹）
extern "C"
{
#include <libavformat/avformat.h>       // 格式上下文、av_read_frame 等
#include <libavcodec/avcodec.h>         // 编解码器参数管理
#include <libavcodec/bsf.h>             // 位流过滤器（H.264 MP4→AnnexB 格式转换）
#include <libavutil/imgutils.h>         // 图像工具函数
#include <libavutil/time.h>             // 时间相关函数
}

#include "data_processing/mpp_decoder.h"    // MPP硬件解码器（Rockchip平台H.264/H.265硬解）

#include <chrono>
#include <thread>
#include <atomic>
#include <functional>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

/**
 * @brief MPP解码器帧回调函数类型
 *
 * 解码器每输出一帧NV12图像时，通过此回调将DMA-BUF文件描述符和图像元数据
 * 传递给上层StreamLoader，实现零拷贝帧传递。
 *
 * @param userdata 用户数据指针（实际指向 StreamLoader 实例）
 * @param width_stride 宽度步长（含硬件对齐填充）
 * @param height_stride 高度步长（含硬件对齐填充）
 * @param width 图像有效宽度（像素）
 * @param height 图像有效高度（像素）
 * @param format 像素格式（仅接受线性NV12: MPP_FMT_YUV420SP）
 * @param fd DMA-BUF文件描述符（跨设备零拷贝共享内存句柄）
 * @param data 图像数据指针（零拷贝链路未使用）
 * @param buffer_size DMA缓冲区总大小（字节）
 * @param id 解码器实例ID
 * @param frame_hold_token 帧生命周期令牌（持有期间MPP不回收该DMA缓冲区）
 */
typedef std::function<void(void *userdata, int width_stride, int height_stride,
                           int width, int height, int format, int fd, void *data,
                           size_t buffer_size, int id,
                           const std::shared_ptr<void>& frame_hold_token)>
    MppDecoderFrameCallback;

/**
 * @class StreamLoader
 * @brief 视频流加载器类
 *
 * 负责单个视频流的打开、读取、解码和帧回调处理。
 * 支持RTSP、RTMP、本地文件等多种视频源。
 */
class StreamLoader
{
public:
    /**
     * @brief 运行时配置选项
     *
     * 通过 configureRuntime() 动态调整网络缓冲、超时、延迟等参数。
     */
    struct RuntimeOptions {
        size_t max_ready_depth = 4;         // 就绪队列最大深度
        int rtbufsize_bytes = 2097152;      // RTSP接收缓冲区大小（字节，默认2MB）
        int stimeout_us = 500000;           // 流超时（微秒，默认500ms）
        int max_delay_us = 100000;          // 最大延迟（微秒，默认100ms）
        std::string rtsp_transport = "tcp"; // RTSP传输协议（tcp更可靠，udp延迟更低）
    };

    /**
     * @brief 硬件解码帧描述符
     *
     * 封装一帧已解码硬件图像的元数据。包含DMA-BUF FD和图像几何信息，
     * 供RGA/NPU等下游模块零拷贝使用。
     */
    struct DecodedHwFrameDesc {
        int dma_fd = -1;                   // DMA-BUF文件描述符（-1=无效）
        int width = 0;                     // 有效宽度（像素）
        int height = 0;                    // 有效高度（像素）
        int width_stride = 0;              // 宽度步长（含对齐填充）
        int height_stride = 0;             // 高度步长（含对齐填充）
        size_t buffer_size = 0;            // DMA缓冲区总大小（字节）
        std::shared_ptr<void> frame_hold;  // 帧生命周期令牌

        bool valid() const {
            return dma_fd > 0 && width > 0 && height > 0;
        }
    };

    // --- 公共成员 ---
    MppDecoder decoder;                    // MPP硬件解码器实例
    int videoStreamIndex;                  // 视频流在AVFormatContext中的索引
    AVDictionary *options = NULL;          // FFmpeg选项字典（网络参数）
    AVFormatContext *fmtCtx = NULL;        // FFmpeg格式上下文
    AVCodecParameters *codecPar = NULL;    // 编解码器参数（编码格式/分辨率等）
    AVBSFContext *bsf_ctx = NULL;          // 位流过滤器上下文（H.264格式转换）
    const AVBitStreamFilter *bsf = nullptr; // 位流过滤器指针（h264_mp4toannexb）

    bool got_key_frame = false;            // 是否已收到关键帧（I帧）
    AVPacket *temp_pkt = nullptr;          // av_read_frame 的临时数据包容器
    int current_pkt_id = 0;                // 当前数据包编号（递增计数）
    int stream_loader_id;                  // 流加载器唯一标识号
    char *stream_url = nullptr;            // 流地址（RTSP URL 或文件路径）
    int width = 0;                         // 视频宽度
    int height = 0;                        // 视频高度
    int status = 0;                        // 状态码（0=正常，非0=出错触发重连）
    bool isnotAnnexB = false;              // 是否需要 MP4→AnnexB 格式转换
    MppDecoderFrameCallback callback;      // 帧回调函数对象

    std::atomic<bool> stopFlag;            // 停止标志（线程安全）

    // ========================================================================
    // 帧消费者 API
    // ========================================================================

    /**
     * @brief 阻塞等待获取一帧已解码的硬件描述符
     *
     * 上层管线的"消费者"入口。调用线程阻塞直到有新帧入队、超时或流停止。
     * 内部使用 condition_variable::wait_for 实现高效等待，无忙轮询。
     *
     * @param[out] out_desc_id 描述符ID（用于后续 releaseDecodedFrame 回收）
     * @param[out] out_hw_desc 硬件帧描述符（DMA FD + 图像几何信息）
     * @param timeout 等待超时（毫秒）
     * @return true=成功获取一帧，false=超时或流已停止
     */
    bool waitAndGetDecodedFrame(uint64_t& out_desc_id,
                                DecodedHwFrameDesc& out_hw_desc,
                                std::chrono::milliseconds timeout);

    /**
     * @brief 释放已消费的帧描述符（归还槽位到回收队列）
     *
     * 消费者处理完一帧后必须调用，将holder从映射表移除，
     * 释放shared_ptr引用计数，允许MPP回收DMA缓冲区。
     */
    void releaseDecodedFrame(uint64_t desc_id);

    /**
     * @brief 解码回调入口（生产者侧）
     *
     * 由 mpp_decoder_frame_callback 调用，将新解码帧推入就绪队列。
     * 内部执行：回收队列排空 → holder map上限保护 → 就绪队列深度控制 → 入队+通知
     */
    void onDecodedFrame(const DecodedHwFrameDesc& hw_desc);

    // ========================================================================
    // 运行时配置与统计
    // ========================================================================

    void configureRuntime(const RuntimeOptions& options);

    size_t readyDepth() const;               // 当前就绪队列帧数
    size_t maxReadyDepth() const;            // 就绪队列最大深度配置值
    std::uint64_t droppedReadyCount() const;  // 因队列满丢弃的帧数
    std::uint64_t droppedHolderCount() const; // 因holder map满丢弃的帧数
    std::uint64_t droppedTotalCount() const;  // 总丢弃帧数

    // ========================================================================
    // 流生命周期
    // ========================================================================

    /**
     * @brief 关闭流并释放所有资源
     *
     * 释放顺序：MPP解码器 → BSF上下文 → AVDictionary → AVPacket →
     * AVFormatContext → AVCodecParameters → 帧描述符队列。
     * 幂等操作，可安全多次调用。
     */
    void close();

    /**
     * @brief 从流中读取一帧并提交硬件解码
     *
     * 完整流水线：av_read_frame → 过滤非视频包 → BSF格式转换(如需要) → MPP解码。
     * 最多重试10次，每次失败后睡眠2ms。
     * @return true=成功提交解码，false=10次重试均失败
     */
    bool read_frame();

    StreamLoader(char *url, int id);
    ~StreamLoader();

    /**
     * @brief 打开并初始化视频流
     *
     * 完整初始化序列：
     * 1. 防御式 close() 清理残留资源
     * 2. 分配 AVPacket / AVCodecParameters
     * 3. 配置FFmpeg网络选项（rtbufsize, stimeout, max_delay, low_delay等）
     * 4. avformat_open_input（RTSP: DESCRIBE→SETUP→PLAY）
     * 5. avformat_find_stream_info（解析SPS/PPS等编解码参数）
     * 6. 定位第一个视频流索引及分辨率
     * 7. 按编码格式初始化MPP硬件解码器（H.264需额外配置BSF过滤器）
     * 8. 绑定解码回调 → 拷贝编解码参数
     *
     * @return 0=成功, -1=FFmpeg分配/打开失败, -2=未找到视频流,
     *         -3=解码器初始化失败, -4=不支持的编码格式, -5=参数复制失败
     */
    int open();

    /**
     * @brief 线程入口函数（operator() 重载）
     *
     * 独立线程的主循环，持续调用 read_frame() 拉流解码。
     *
     * 线程模型：单生产者线程（此operator()）持续从网络拉流并提交MPP硬解。
     * MPP解码完成后通过回调（mpp_decoder_frame_callback）异步通知，
     * 将帧描述符推入 ready_queue_。消费者线程通过 waitAndGetDecodedFrame 取帧。
     *
     * 自动重连策略（指数退避）：
     * - 检测 status != 0 → close() → open() 循环重试
     * - 退避序列：1s → 2s → 4s → 8s → 15s → 30s（上限）
     * - 退避期间每100ms检查 stopFlag，确保快速响应停止信号
     * - 退出时广播 notify_all() 唤醒所有等待消费者
     */
    void operator()();

private:
    /** @brief 解码帧槽位，封装一帧的完整描述符 */
    struct DecodedFrameSlot {
        DecodedHwFrameDesc hw_desc;
    };

    /**
     * @brief 排空待回收队列（调用前必须已持有 desc_mutex_ 锁）
     *
     * 遍历 recycle_queue_，逐一从 out_desc_to_holder_ 映射表中移除条目，
     * 释放 shared_ptr 引用以允许MPP回收DMA缓冲区。
     */
    void drainPendingRecycleQueueLocked();

    // ========================================================================
    // 帧管理数据结构 — 生产者-消费者模型核心
    // ========================================================================
    mutable std::mutex desc_mutex_;                                  // 保护以下所有成员的互斥锁
    std::condition_variable desc_cv_;                                // 消费者等待条件变量
    std::deque<uint64_t> ready_queue_;                               // 就绪队列（FIFO，存放desc_id）
    std::deque<uint64_t> recycle_queue_;                             // 回收队列（延迟回收desc_id）
    std::unordered_map<uint64_t, std::shared_ptr<DecodedFrameSlot>>
        out_desc_to_holder_;                                         // desc_id → 帧槽位映射(O(1)查找)
    uint64_t next_desc_id_ = 1;                                      // 单调递增描述符ID生成器
    size_t max_ready_depth_ = 4;                                     // 就绪队列最大深度
    std::uint64_t dropped_ready_count_ = 0;                          // 队列满丢弃计数
    std::uint64_t dropped_holder_count_ = 0;                         // holder表满丢弃计数

    // ========================================================================
    // 网络流运行时参数
    // ========================================================================
    int rtbufsize_bytes_ = 1048576;        // 接收缓冲区（默认1MB）
    int stimeout_us_ = 2000000;            // 流超时（默认2s）
    int max_delay_us_ = 100000;            // 最大延迟（默认100ms）
    std::string rtsp_transport_ = "tcp";   // RTSP传输协议
};
