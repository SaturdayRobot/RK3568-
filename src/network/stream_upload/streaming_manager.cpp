#include "network/stream_upload/streaming_manager.h"  // 流管理器类声明
#include "data_processing/mpp_encoder.h"              // RK MPP 原生 H.264 硬件编码器
#include "pipeline/rga_preprocessor.h"                // RGA 2D 硬件加速预处理（格式转换+缩放）
#include <algorithm>       // std::max, std::min 等
#include <chrono>          // 时间库：时间点、间隔测量
#include <cstring>         // C 字符串操作
#include <iostream>        // 标准输入输出：日志和错误输出
#include <utility>         // std::move 移动语义

#include "utils/thread_runtime.h"  // 线程实时调优（绑核、调度策略）

extern "C" {
#include <libavformat/avformat.h>   // 封装格式上下文，处理RTSP协议
#include <libavcodec/avcodec.h>     // 编解码核心接口
#include <libavutil/avutil.h>       // 通用工具函数（错误处理、时间基转换）
#include <libavutil/imgutils.h>     // 图像数据处理（帧缓冲区分配）
#include <libswscale/swscale.h>     // 图像格式/尺寸转换（CPU降级方案）
#include <libavutil/opt.h>          // 编码器参数设置
}

// 匿名命名空间：仅当前编译单元可见，避免全局符号冲突
namespace {
/**
 * @brief 安全计算帧间隔时间（防止除零错误）
 * @param fps 目标帧率（帧/秒）
 * @return 帧间隔毫秒数
 * @note 当 fps <= 0 时，默认使用 1fps（即 1000ms 间隔）
 */
int safeFrameIntervalMs(int fps) {
    // 确保分母不为0，fps<=0时默认1fps（1000ms间隔）
    return 1000 / ((fps > 0) ? fps : 1);  // 整数除法取整
}

/**
 * @brief 关闭输出流并释放所有 FFmpeg 相关资源
 *
 * 执行流程：
 *   1. 编码器收尾：发送空帧触发编码器输出剩余包
 *   2. 写入流尾信息（RTSP 必须，否则接收端无法正常结束播放）
 *   3. 关闭 IO 上下文
 *   4. 释放编码器上下文和格式上下文
 *
 * @param fmt_ctx_ptr   输出格式上下文二级指针（释放后置空）
 * @param codec_ctx_ptr 编码器上下文二级指针（释放后置空）
 */
void closeOutput(AVFormatContext** fmt_ctx_ptr, AVCodecContext** codec_ctx_ptr) {
    // 解引用指针，简化后续操作
    AVFormatContext* fmt_ctx = fmt_ctx_ptr ? *fmt_ctx_ptr : nullptr;    // 格式上下文
    AVCodecContext* codec_ctx = codec_ctx_ptr ? *codec_ctx_ptr : nullptr; // 编码器上下文
    // 无有效资源直接返回
    if (!fmt_ctx && !codec_ctx) {
        return;
    }

    // --- 步骤 1：编码器收尾 ---
    // 编码器可能有内部缓冲数据尚未输出，需发送空帧触发冲刷
    if (codec_ctx && fmt_ctx && fmt_ctx->streams && fmt_ctx->nb_streams > 0) {
        // 发送NULL帧告知编码器结束编码，触发内部缓冲冲刷
        avcodec_send_frame(codec_ctx, nullptr);
        // 分配数据包对象，接收编码器剩余数据
        AVPacket* pkt = av_packet_alloc();
        // 循环接收编码器输出的数据包，直到返回 < 0（无更多数据）
        while (avcodec_receive_packet(codec_ctx, pkt) >= 0) {
            pkt->stream_index = 0;  // 指定流索引（本管理器仅包含单路视频流）
            // 转换时间基：编码器时间基 -> 封装格式时间基（保证 PTS/DTS 正确）
            av_packet_rescale_ts(pkt, codec_ctx->time_base, fmt_ctx->streams[0]->time_base);
            // 交错写入数据包（保证 RTSP 流中视频帧时序正确）
            av_interleaved_write_frame(fmt_ctx, pkt);
            // 重置数据包（释放数据缓冲区引用，保留对象本身用于下一次循环）
            av_packet_unref(pkt);
        }
        // 释放数据包对象
        av_packet_free(&pkt);
        // 写入流尾信息（RTSP 必需，否则流无法正常结束，播放器会卡住）
        av_write_trailer(fmt_ctx);
    }

    // --- 步骤 2：关闭 IO 上下文 ---
    // 仅对非内存流的输出格式关闭 IO（RTSP 需要关闭网络连接）
    if (fmt_ctx && !(fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        avio_closep(&fmt_ctx->pb);  // 关闭并释放 AVIOContext（关闭网络连接）
    }

    // --- 步骤 3：释放编码器上下文 ---
    if (codec_ctx_ptr && *codec_ctx_ptr) {
        avcodec_free_context(codec_ctx_ptr);   // 释放编解码器上下文，同时置空指针
    }

    // --- 步骤 4：释放格式上下文 ---
    // 释放封装格式上下文及其内部资源，并置空指针避免野指针
    if (fmt_ctx_ptr && *fmt_ctx_ptr) {
        avformat_free_context(*fmt_ctx_ptr);
        *fmt_ctx_ptr = nullptr;                // 置空调用方的指针
    }
}
} // namespace

/**
 * @file streaming_manager.cpp
 * @brief 视频推流管理器实现
 * @details 实现 StreamingManager 类，负责视频流推流管理，
 *          支持 RTSP 推流，包含视频编码、推流与断线重连。
 *          四条视频管线（Visible / VisibleRtsp / Infrared / Mosaic）
 *          的输出统一收敛到此类，由 RTSP worker 线程编码发送。
 */

/**
 * @brief 构造函数：初始化所有成员变量为安全默认值
 *
 * 成员初始化列表对齐声明顺序，保证：
 *   - 线程安全：所有 atomic 变量显式初始化为 false/0
 *   - 指针安全：所有 void* 置空避免悬垂
 */
StreamingManager::StreamingManager()
    : streaming_active_(false)          // 推流状态：未激活
    , should_stop_(false)               // 停止标志：未停止
    , avformat_context_rtsp_(nullptr)   // RTSP 封装上下文：空
    , avcodec_context_rtsp_(nullptr)    // RTSP 编码器上下文：空
    , sws_context_cache_(nullptr)       // 图像转换上下文缓存：空
    , av_frame_cache_(nullptr)          // AVFrame 缓存：空
    , av_packet_cache_(nullptr)         // AVPacket 缓存：空
    , mpp_encoder_(nullptr)             // 原生 MPP 编码器：空
    , frame_count_(0)                   // 帧计数：0
    , rtsp_frame_index_(0)              // RTSP 帧序号：0
    , consecutive_send_failures_(0)     // 连续发送失败次数：0
{
}

/**
 * @brief 析构函数：停止流管理器并释放全部资源
 *
 * 调用 stopStreaming() 执行完整的资源释放流程：
 *   1. 通知工作线程退出
 *   2. 等待线程 join
 *   3. 清空队列
 *   4. 释放 FFmpeg/RGA/MPP 资源
 */
StreamingManager::~StreamingManager() {
    stopStreaming();  // 析构时确保停止推流、释放资源
}

/**
 * @brief 初始化流管理器
 *
 * StreamingManager 是所有视频管线共用的"最终外发器"。
 * 上游不管来自 visible / rtsp / infrared / mosaic，走到这里都必须被归一成：
 *   1. BGR CV_8UC3 Mat 输入
 *   2. 固定宽高（与编码器配置匹配）
 *   3. 统一交给 RTSP worker 线程编码发送
 *
 * @param config 流配置参数（包含 RTSP 地址、分辨率、帧率、码率等）
 * @return 初始化成功返回 true，失败返回 false
 */
bool StreamingManager::initialize(const StreamingConfig& config) {
    config_ = config;  // 保存配置参数副本
    if (config_.max_queue_size <= 0) {
        config_.max_queue_size = kMaxQueueSize;  // 队列深度为 0 时使用默认值
    }

    // StreamingManager 是所有视频管线共用的"最终外发器"。
    // 上游不管来自 visible / rtsp / infrared / mosaic，走到这里都必须被归一成：
    // 1. BGR Mat 输入
    // 2. 固定宽高
    // 3. 统一交给 RTSP worker 线程编码发送

    // 启用 RTSP 时初始化 FFmpeg 编码链路（含 MPP 硬件编码器探测）
    if (config_.enable_rtsp) {
        if (!initializeRTSP()) {          // 初始化 RTSP 输出（编码器 + 格式上下文）
            std::cerr << "Failed to initialize RTSP streaming" << std::endl;
            return false;                 // 初始化失败，向上报告
        }
    }

    return true;  // 初始化成功
}

/**
 * @brief 添加流数据到推流队列
 *
 * 队列生产者接口，由各视频管线的处理线程调用。
 * 队列满时的处理策略：
 *   - drop_oldest_when_full=true：丢弃最旧帧（弹队首）
 *   - drop_oldest_when_full=false：丢弃当前新帧（直接返回）
 *
 * @param data 待推流的帧数据及检测结果
 * @note 使用移动语义避免拷贝开销，添加后唤醒推流工作线程
 */
void StreamingManager::addStreamingData(StreamingData data) {
    std::lock_guard<std::mutex> lock(queue_mutex_);  // 加锁：保证队列线程安全
    const std::size_t max_q = static_cast<std::size_t>(std::max(1, config_.max_queue_size)); // 队列最大容量

    // 队列满时的丢帧策略：控制队列长度，降低端到端延迟
    while (streaming_queue_.size() >= max_q) {  // 循环弹出直到队列未满
        if (!config_.drop_oldest_when_full) {   // 策略：丢弃新帧
            std::lock_guard<std::mutex> slock(stats_mutex_);  // 独立锁统计量
            stats_.frames_dropped++;            // 递增丢帧计数器
            return;                             // 直接丢弃，不加入队列
        }
        streaming_queue_.pop();                 // 策略：丢弃最旧帧（弹队首）
        std::lock_guard<std::mutex> slock(stats_mutex_);
        stats_.frames_dropped++;                // 递增丢帧计数器
    }

    // 移动语义添加数据（避免深拷贝 cv::Mat 的大量像素数据）
    streaming_queue_.push(std::move(data));
    queue_cv_.notify_one();  // 唤醒推流线程：有新数据到达，可以立即处理
}

/**
 * @brief 启动推流线程
 *
 * 创建独立工作线程运行 streamingWorker()，负责：
 *   - 从队列取帧
 *   - BGR→NV12 转换（RGA 硬件加速）
 *   - MPP 硬件编码
 *   - RTSP 网络发送
 */
void StreamingManager::startStreaming() {
    // 已激活则直接返回（避免重复启动导致多线程竞争）
    if (streaming_active_.load()) {
        return;
    }

    streaming_active_ = true;   // 标记推流激活
    should_stop_ = false;       // 重置停止标志
    // 启动推流工作线程，std::thread 立即开始执行
    streaming_thread_ = std::thread(&StreamingManager::streamingWorker, this);

    std::cout << "Streaming started" << std::endl;
}

/**
 * @brief 停止推流线程，释放所有资源
 *
 * 完整的资源释放流程：
 *   1. 设置停止标志 should_stop_
 *   2. 唤醒条件变量，让工作线程从 wait_for 中退出
 *   3. join 等待线程结束
 *   4. 清空队列，释放帧内存
 *   5. 释放 RTSP 输出资源（FFmpeg 格式上下文 + 编码器上下文）
 *   6. 释放 MPP 编码器
 *   7. 释放缓存：SwsContext、AVFrame、AVPacket
 */
void StreamingManager::stopStreaming() {
    // 未激活则直接返回（幂等操作）
    if (!streaming_active_.load()) {
        return;
    }

    should_stop_ = true;        // 设置停止标志，通知工作线程退出主循环
    queue_cv_.notify_all();     // 唤醒所有等待的线程（包括可能在 wait_for 中的 streamingWorker）

    // 等待工作线程退出（阻塞直到线程函数返回）
    if (streaming_thread_.joinable()) {
        streaming_thread_.join();
    }

    streaming_active_ = false;  // 标记推流未激活

    // 清空残留队列，释放帧内存（避免内存泄漏）
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        std::queue<StreamingData> empty;     // 创建空队列
        streaming_queue_.swap(empty);        // 交换，原队列在 swap 后析构（自动释放 Mat 内存）
    }

    // 关闭 RTSP 输出，释放 FFmpeg 格式上下文和编码器上下文资源
    closeOutput(reinterpret_cast<AVFormatContext**>(&avformat_context_rtsp_),
                reinterpret_cast<AVCodecContext**>(&avcodec_context_rtsp_));
    // 释放 RK MPP 原生 H.264 硬件编码器
    if (mpp_encoder_) {
        delete static_cast<MppH264Encoder*>(mpp_encoder_); // 调用析构函数释放 MPP 资源
        mpp_encoder_ = nullptr;                             // 置空
    }

    // 释放缓存的 SwsContext（CPU 格式转换上下文）
    if (sws_context_cache_) {
        sws_freeContext(static_cast<SwsContext*>(sws_context_cache_));
        sws_context_cache_ = nullptr;       // 置空
    }
    // 释放缓存的 AVFrame（编码前帧缓冲区）
    if (av_frame_cache_) {
        AVFrame* cached = static_cast<AVFrame*>(av_frame_cache_);
        av_frame_free(&cached);             // 释放帧对象及其内部缓冲区
        av_frame_cache_ = nullptr;
    }
    // 释放缓存的 AVPacket（编码后数据包对象）
    if (av_packet_cache_) {
        AVPacket* cached_pkt = static_cast<AVPacket*>(av_packet_cache_);
        av_packet_free(&cached_pkt);        // 释放包对象
        av_packet_cache_ = nullptr;
    }

    std::cout << "Streaming stopped" << std::endl;
}

/**
 * @brief 推流工作线程主函数
 *
 * 核心处理流程（每一轮循环）：
 *   1. 从队列抓取帧数据（阻塞等待，超时触发补帧）
 *   2. 补帧机制：无新帧时复用上一帧，防止播放器因断流跳出
 *   3. 分辨率校验与自动 resize（RGA 硬件 + CPU 回退）
 *   4. 更新帧缓存（供下一周期补帧）
 *   5. 执行编码与发送：BGR → NV12 → RKMPP 编码 → RTSP 网络发送
 *   6. 容错与重连：连续失败触发 RTSP 重连
 *   7. 统计信息更新（FPS 计算）
 *
 * 线程被 stopStreaming() 设置 should_stop_ 标志后安全退出。
 */
void StreamingManager::streamingWorker() {
    utils::applyThreadRuntime("streaming", "streaming");  // 应用实时线程调优配置

    cv::Mat last_frame;             // 上一帧缓存（用于补帧时复用）
    bool has_last_frame = false;    // 是否有有效缓存帧（刚启动时无缓存）
    // 计算帧间隔（控制推流帧率，保证 RTSP 输出平稳）
    const auto frame_interval = std::chrono::milliseconds(safeFrameIntervalMs(config_.fps));
    const int fps_for_stats = (config_.fps > 0) ? config_.fps : 1;  // 统计用帧率（防止除零）

    // 主循环：未收到停止指令则持续运行
    while (!should_stop_.load()) {
        StreamingData data;         // 帧数据容器
        bool got_data = false;      // 本轮是否获取到有效数据

        // === 第一步：从队列抓取数据（生产者-消费者交接） ===
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);  // 上锁
            // 阻塞等待，直到队列有数据、收到停止信号或超时（用于补帧判定）
            if (queue_cv_.wait_for(lock, frame_interval, [this] {
                return !streaming_queue_.empty() || should_stop_.load();  // 唤醒条件
            })) {
                if (should_stop_.load()) break;  // 收到停止信号，立即退出

                // 核心逻辑：若队列积压，只取最新一帧（跳过旧帧以维持实时性）
                if (!streaming_queue_.empty()) {
                    if (config_.consume_latest_only) {          // 启用"仅取最新帧"模式
                        while (streaming_queue_.size() > 1) {   // 保留最后一帧
                            streaming_queue_.pop();             // 丢弃中间旧帧
                            std::lock_guard<std::mutex> slock(stats_mutex_);
                            stats_.frames_dropped++;            // 计入丢帧统计
                        }
                    }
                    data = std::move(streaming_queue_.front()); // 移动语义取队首帧
                    streaming_queue_.pop();                     // 出队
                    got_data = true;                            // 成功获取数据
                }
            } else {
                // === 第二步：补帧机制（流稳定性保护） ===
                // 如果等待超时仍无新图（生产者断流），复用上一帧图推流，
                // 防止播放器因长时间收不到数据而断流跳出
                if (has_last_frame && !last_frame.empty()) {
                    data.frame = last_frame;                    // 复用缓存帧
                    data.stream_id = -1;                        // 标记为补帧（特殊 stream_id）
                    data.timestamp = std::chrono::system_clock::now(); // 使用当前时间戳
                    got_data = true;                            // 获取到补帧数据
                } else {
                    continue;                                   // 无缓存帧，继续等待
                }
            }
        }

        if (!got_data) continue;  // 未获取到任何帧，跳过本轮

        // === 第三步：数据预校验（防止编码器崩溃） ===
        // 校验分辨率是否与初始化时的编码器配置一致
        const bool need_resize = (data.frame.cols != config_.width ||   // 宽度不匹配
                                  data.frame.rows != config_.height);    // 高度不匹配
        if (need_resize) {
            if (!config_.resize_on_mismatch) {                  // 不允许自动 resize
                std::cerr << "[StreamingManager] frame size mismatch, drop frame" << std::endl;
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.frames_dropped++;                        // 丢弃该帧
                continue;
            }

            // 使用 RGA 硬件 resize 替代 CPU cv::resize（大幅降低 CPU 占用）
            cv::Mat resized;                                    // resize 目标图像
            {
                thread_local pipeline::RgaPreprocessor rga_resizer;     // 线程局部 RGA 实例
                thread_local bool resizer_initialized = false;          // 初始化标志
                if (!resizer_initialized) {                             // 首次使用时初始化
                    pipeline::RgaPreprocessConfig cfg;
                    cfg.use_rga = true;                                 // 启用 RGA 硬件
                    cfg.target_width = config_.width;                   // 目标宽度
                    cfg.target_height = config_.height;                 // 目标高度
                    cfg.src_format = pipeline::RgaPixelFormat::BGR888;  // 源格式 BGR
                    cfg.dst_format = pipeline::RgaPixelFormat::BGR888;  // 目标格式 BGR（仅缩放）
                    rga_resizer.initialize(cfg);                        // 初始化 RGA
                    resizer_initialized = true;                         // 标记已初始化
                }
                if (!rga_resizer.process(data.frame, resized)) {        // RGA 硬件缩放
                    // RGA 失败时回退到 CPU resize（OpenCV 实现）
                    cv::resize(data.frame, resized, cv::Size(config_.width, config_.height),
                               0.0, 0.0, cv::INTER_LINEAR);            // 双线性插值
                }
            }
            data.frame = std::move(resized);  // 替换为 resize 后的帧
        }

        // === 第四步：更新缓存与所有权接管 ===
        cv::Mat frame = std::move(data.frame);  // 转移所有权（避免深拷贝）
        last_frame = frame;                      // 存入缓存供下一周期的"补帧"使用
        has_last_frame = true;                   // 标记有有效缓存

        // === 第五步：推送执行（核心编码与网络发送） ===
        // 到这里说明这帧已经通过了尺寸/类型校验，后面就是：
        // BGR -> NV12 -> RKMPP 编码 -> RTSP 发送。
        bool success = false;                    // 发送结果标志
        if (config_.enable_rtsp) {
            // 调用 FFmpeg + RKMPP 硬件链路发送（格式转换+编码+推流一体化）
            success = sendRTSPFrame(frame);      // 核心编码发送函数
        }

        // === 第六步：容错与重连处理（增强鲁棒性） ===
        if (config_.enable_rtsp) {
            if (success) {
                consecutive_send_failures_ = 0;  // 成功则重置连续失败计数器
            } else {
                consecutive_send_failures_++;     // 失败则递增计数
                // 若连续失败次数超限，通常是网络断开，尝试触发 FFmpeg 重新握手
                if (consecutive_send_failures_ >= kMaxConsecutiveFailures) {
                    std::cerr << "[StreamingManager] Reconnecting RTSP..." << std::endl;
                    if (tryReconnectRTSP()) {     // 尝试重连
                        std::cout << "[StreamingManager] Reconnect succeeded" << std::endl;
                    }
                    consecutive_send_failures_ = 0;  // 重置计数器（无论重连成功与否）
                }
            }
        }

        // === 第七步：统计分析（性能监控） ===
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);  // 统计锁
            if (success) stats_.frames_sent++;   // 成功：递增发送计数
            else stats_.frames_dropped++;         // 失败：递增丢帧计数

            // 实时 FPS 采样计算：每 fps_for_stats 帧计算一次
            auto now = std::chrono::system_clock::now();     // 当前时刻
            frame_count_++;                                   // 帧计数递增
            if (frame_count_ % fps_for_stats == 0) {          // 达到采样间隔
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - stats_.last_frame_time).count();     // 时间差（ms）
                if (duration > 0) {                            // 避免除零
                    stats_.fps = (fps_for_stats * 1000.0) / duration; // FPS = 帧数 / 秒
                }
                stats_.last_frame_time = now;                  // 更新最后一次采样时间
            }
        }
    }
}

/**
 * @brief 初始化 RTSP 推流（使用 RKMPP 硬件编码器优先）
 *
 * 编码器探测优先级：
 *   1. 原生 MPP H.264 编码器（MppH264Encoder）—— 最优路径，零拷贝
 *   2. h264_rkmpp  —— FFmpeg 封装的 MPP 编码器
 *   3. h264_omx     —— OpenMAX 硬件编码器
 *   4. h264_v4l2m2m —— V4L2 内存到内存编码器
 *   5. libx264      —— CPU 软件编码器（最终兜底）
 *
 * @return 初始化成功返回 true，失败返回 false
 */
bool StreamingManager::initializeRTSP() {
    avformat_network_init();  // 初始化 FFmpeg 网络模块（RTSP 协议栈必需）

    AVFormatContext* fmt_ctx = nullptr;      // FFmpeg 封装格式上下文
    AVCodecContext* codec_ctx = nullptr;     // FFmpeg 编码器上下文
    const AVCodec* codec = nullptr;          // FFmpeg 编码器描述

    // === 尝试路径1：原生 MPP H.264 编码器（零拷贝，最优性能） ===
    // 创建 RTSP 输出格式上下文（FFmpeg 自动识别 rtsp:// 协议前缀）
    avformat_alloc_output_context2(&fmt_ctx, nullptr, "rtsp", config_.rtsp_url.c_str());
    if (!fmt_ctx) {
        std::cerr << "Could not create RTSP output context" << std::endl;
        return false;                        // 创建格式上下文失败
    }

    const int fps = (config_.fps > 0) ? config_.fps : 25;  // 默认 25fps
    auto* native_mpp = new MppH264Encoder();                // 创建原生 MPP 编码器
    if (native_mpp->initialize(config_.width, config_.height, fps, config_.bitrate)) { // 初始化编码器
        AVStream* stream = avformat_new_stream(fmt_ctx, nullptr);  // 创建视频流
        if (stream) {
            stream->time_base = {1, fps};                // 设置时间基（1/fps）
            stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO; // 流类型：视频
            stream->codecpar->codec_id = AV_CODEC_ID_H264;     // 编码格式：H.264
            stream->codecpar->width = config_.width;           // 视频宽度
            stream->codecpar->height = config_.height;         // 视频高度
            const auto& extra = native_mpp->header();          // 获取 H.264 SPS/PPS 头信息
            // 分配 extradata 缓冲区（包含 AV_INPUT_BUFFER_PADDING_SIZE 的末尾填充，满足 FFmpeg 安全要求）
            stream->codecpar->extradata = static_cast<uint8_t*>(
                av_mallocz(extra.size() + AV_INPUT_BUFFER_PADDING_SIZE));
            if (stream->codecpar->extradata) {
                std::memcpy(stream->codecpar->extradata, extra.data(), extra.size()); // 拷贝 SPS/PPS
                stream->codecpar->extradata_size = static_cast<int>(extra.size());    // 设置大小
            }
            AVDictionary* opts = nullptr;               // RTSP 选项字典
            av_dict_set(&opts, "rtsp_transport", "tcp", 0);  // 使用 TCP 传输（可靠，但延迟略高）
            av_dict_set(&opts, "muxdelay", "0", 0);          // 最小化封装延迟（实时推流要求）
            int ret = 0;

            // 打开 RTSP IO（网络输出）
            if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE))
                ret = avio_open2(&fmt_ctx->pb, config_.rtsp_url.c_str(),
                                 AVIO_FLAG_WRITE, nullptr, &opts);  // 打开 RTSP 输出
            if (ret >= 0) ret = avformat_write_header(fmt_ctx, &opts); // 写入流头
            av_dict_free(&opts);                        // 释放选项字典
            if (ret >= 0) {
                avformat_context_rtsp_ = fmt_ctx;       // 保存格式上下文
                avcodec_context_rtsp_ = nullptr;        // 原生 MPP 路径不需要 FFmpeg 编码器上下文
                mpp_encoder_ = native_mpp;              // 保存 MPP 编码器指针
                std::cout << "[StreamingManager] H.264 encoder opened: native_mpp\n";
                return true;                            // 初始化成功
            }
            if (fmt_ctx->pb) avio_closep(&fmt_ctx->pb); // 失败时关闭 IO
        }
        std::cerr << "[StreamingManager] native MPP mux setup failed, using fallback\n";
    }
    delete native_mpp;  // 原生 MPP 路径失败，释放编码器

    // 如果前面已创建了流，需要重新创建格式上下文（避免状态污染）
    if (fmt_ctx->nb_streams > 0) {
        avformat_free_context(fmt_ctx);     // 释放旧上下文
        fmt_ctx = nullptr;
        avformat_alloc_output_context2(&fmt_ctx, nullptr, "rtsp", config_.rtsp_url.c_str()); // 重新创建
        if (!fmt_ctx) return false;
    }

    // === 尝试路径2-5：FFmpeg 硬件/软件编码器候选列表 ===
    const char* encoder_name = nullptr;     // 最终选中的编码器名称
    const char* encoder_candidates[] = {    // 按优先级排序的编码器候选
        "h264_rkmpp",     // RKMPP 硬件编码器（FFmpeg 封装）
        "h264_omx",       // OpenMAX 硬件编码器
        "h264_v4l2m2m",   // V4L2 内存到内存编码器
        "libx264"         // CPU 软件编码器（兜底方案）
    };
    for (const char* candidate : encoder_candidates) {    // 依次尝试
        const AVCodec* candidate_codec = avcodec_find_encoder_by_name(candidate); // 查找编码器
        if (!candidate_codec) {
            std::cerr << "[StreamingManager] encoder unavailable: " << candidate << std::endl;
            continue;                                   // 该编码器不可用，尝试下一个
        }
        AVCodecContext* candidate_ctx = avcodec_alloc_context3(candidate_codec); // 分配编码器上下文
        if (!candidate_ctx) continue;

        // 配置编码器参数
        candidate_ctx->bit_rate = config_.bitrate;          // 目标码率（bps）
        candidate_ctx->width = config_.width;               // 视频宽度
        candidate_ctx->height = config_.height;              // 视频高度
        candidate_ctx->time_base = {1, fps};                 // 编码器时间基
        candidate_ctx->framerate = {fps, 1};                 // 帧率（用分数表示，精确）
        candidate_ctx->gop_size = 10;                        // GOP 大小（关键帧间隔）
        candidate_ctx->max_b_frames = 0;                     // B 帧数量（实时推流禁用 B 帧降低延迟）
        candidate_ctx->pix_fmt = AV_PIX_FMT_NV12;            // 像素格式 NV12（RK 硬件编码器原生格式）
        if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)    // 全局头标志
            candidate_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; // 编码器输出全局头信息

        // libx264 特殊调优：zerolatency 模式适合实时推流
        if (std::strcmp(candidate, "libx264") == 0) {
            av_opt_set(candidate_ctx->priv_data, "preset", "veryfast", 0);    // 编码速度优先
            av_opt_set(candidate_ctx->priv_data, "tune", "zerolatency", 0);  // 零延迟优化
        }

        const int open_ret = avcodec_open2(candidate_ctx, candidate_codec, nullptr); // 打开编码器
        if (open_ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE]{};    // 错误信息缓冲区
            av_strerror(open_ret, errbuf, sizeof(errbuf)); // 获取可读错误信息
            std::cerr << "[StreamingManager] encoder open failed: " << candidate
                      << " (" << errbuf << "), trying next" << std::endl;
            avcodec_free_context(&candidate_ctx);       // 释放上下文
            continue;                                   // 尝试下一个编码器
        }
        codec = candidate_codec;                        // 保存编码器描述
        codec_ctx = candidate_ctx;                      // 保存编码器上下文
        encoder_name = candidate;                       // 保存编码器名称
        break;                                          // 找到可用编码器，跳出循环
    }
    if (!codec || !codec_ctx) {                         // 所有编码器都不可用
        std::cerr << "No usable H.264 encoder found (tried rkmpp, omx, v4l2m2m, libx264)" << std::endl;
        avformat_free_context(fmt_ctx);                 // 释放格式上下文
        return false;                                   // 初始化失败
    }
    std::cout << "[StreamingManager] H.264 encoder opened: " << encoder_name << std::endl;

    // 创建视频流
    AVStream* stream = avformat_new_stream(fmt_ctx, codec);  // 在格式上下文中创建新流
    if (!stream) {
        std::cerr << "Could not create stream" << std::endl;
        avcodec_free_context(&codec_ctx);               // 释放编码器上下文
        avformat_free_context(fmt_ctx);                 // 释放格式上下文
        return false;
    }

    // 设置流时间基（与编码器一致，避免 PTS/DTS 转换误差）
    stream->time_base = codec_ctx->time_base;
    // 将编码器参数拷贝到流参数（FFmpeg 自动复制 codecpar）
    avcodec_parameters_from_context(stream->codecpar, codec_ctx);

    // 设置 RTSP 推流参数
    AVDictionary* opts_storage = nullptr;                               // 选项字典
    av_dict_set(&opts_storage, "rtsp_transport", "tcp", 0);            // 使用 TCP 传输（可靠，避免 UDP 丢包）
    av_dict_set(&opts_storage, "muxdelay", "0", 0);                    // 最小化封装延迟（实时推流关键参数）

    // 打开 RTSP IO 上下文
    if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open2(&fmt_ctx->pb, config_.rtsp_url.c_str(),
                       AVIO_FLAG_WRITE, nullptr, &opts_storage) < 0) { // 打开 RTSP 网络输出
            std::cerr << "Could not open RTSP output" << std::endl;
            avcodec_free_context(&codec_ctx);           // 释放编码器
            avformat_free_context(fmt_ctx);             // 释放格式上下文
            return false;
        }
    }

    // 写入 RTSP 流头信息（必需，否则接收端无法解析流）
    if (avformat_write_header(fmt_ctx, &opts_storage) < 0) {
        std::cerr << "Error occurred when opening RTSP output" << std::endl;
        if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&fmt_ctx->pb);                // 关闭网络 IO
        }
        avcodec_free_context(&codec_ctx);             // 释放编码器
        avformat_free_context(fmt_ctx);               // 释放格式上下文
        return false;
    }

    // 保存上下文指针供 sendRTSPFrame() 使用
    avformat_context_rtsp_ = fmt_ctx;   // 格式上下文
    avcodec_context_rtsp_ = codec_ctx;  // 编码器上下文
    std::cout << "RTSP encoder: " << (codec->name ? codec->name : "unknown") << std::endl;
    return true;  // 初始化成功
}

/**
 * @brief 发送一帧图像到 RTSP 流（核心编码发送逻辑）
 *
 * 完整的数据处理流水线：
 *   1. 基础校验（上下文、帧格式）
 *   2. 类型还原（void* → FFmpeg 原生类型）
 *   3. 颜色空间转换：BGR → NV12（优先 RGA 硬件，回退 Swscale CPU）
 *   4. 帧属性设置与 DMA 对齐
 *   5. 送入编码器（avcodec_send_frame）
 *   6. 接收编码包（avcodec_receive_packet）
 *   7. PTS/DTS 时间基转换
 *   8. RTSP 网络写入（av_interleaved_write_frame）
 *
 * @param frame OpenCV BGR CV_8UC3 格式的输入帧
 * @return 发送成功返回 true，失败返回 false
 */
bool StreamingManager::sendRTSPFrame(const cv::Mat& frame) {
    // === 步骤 1：基础上下文校验 ===
    // 确保 FFmpeg 编码器和封装上下文已准备就绪
    if (!avformat_context_rtsp_ || (!avcodec_context_rtsp_ && !mpp_encoder_)) {
        return false;  // 任一上下文缺失，返回失败
    }

    // 公共前提保护：StreamingManager 当前统一接收 BGR CV_8UC3 图像。
    // Visible / VisibleRtsp / Infrared / Mosaic 四条管线都遵守这一约定；
    // 若未来有新管线误传其它格式，直接拒绝并保留日志，避免在 RGA/FFmpeg 中走到未定义行为。
    if (frame.empty() || frame.type() != CV_8UC3) {      // 必须是非空的 BGR 三通道图像
        static thread_local uint64_t invalid_frame_count = 0;  // 线程局部无效帧计数
        if (++invalid_frame_count % 100 == 1) {           // 每 100 帧报告一次，避免日志刷屏
            std::cerr << "[StreamingManager] unsupported frame type for streaming: type="
                      << frame.type() << ", channels=" << frame.channels()
                      << ", empty=" << frame.empty()
                      << " (expect CV_8UC3/BGR)" << std::endl;
        }
        return false;
    }

    // === 步骤 2：类型转换 ===
    // 将私有成员中的 void* 句柄还原为 FFmpeg 原生上下文类型
    AVFormatContext* fmt_ctx = static_cast<AVFormatContext*>(avformat_context_rtsp_); // 格式上下文

    // ---- 原生 MPP 编码器路径（零拷贝，最优性能） ----
    if (mpp_encoder_) {
        auto* encoder = static_cast<MppH264Encoder*>(mpp_encoder_);   // 还原 MPP 编码器指针
        static thread_local pipeline::RgaPreprocessor native_rga;     // 线程局部 RGA 实例
        static thread_local bool native_rga_ready = false;            // RGA 就绪标志
        if (!native_rga_ready) {                                      // 首次初始化 RGA
            pipeline::RgaPreprocessConfig cfg;
            cfg.use_rga = true;                                         // 启用 RGA
            cfg.strict_hardware = true;                                 // 严格硬件模式（不可用即失败）
            cfg.target_width = config_.width;                           // 目标宽度
            cfg.target_height = config_.height;                         // 目标高度
            cfg.src_format = pipeline::RgaPixelFormat::BGR888;          // 源格式 BGR
            cfg.dst_format = pipeline::RgaPixelFormat::NV12;            // 目标格式 NV12（MPP 原生格式）
            native_rga_ready = native_rga.initialize(cfg) && native_rga.isRgaActive(); // 初始化并检查可用性
        }
        // 使用 RGA 将 BGR 帧直接写入 MPP 编码器的输入缓冲区（零拷贝，BGR→NV12）
        if (!native_rga_ready || !encoder->inputData() ||
            !native_rga.processToBuffer(frame, encoder->inputData(), encoder->stride(),
                                        encoder->verticalStride()))
            return false;  // RGA 或编码器未就绪，返回失败

        std::vector<uint8_t> encoded;   // 编码后的 H.264 数据
        bool key_frame = false;         // 是否为关键帧（I 帧）
        const int64_t pts = rtsp_frame_index_++;  // 帧 PTS（单调递增）
        if (!encoder->encode(pts, encoded, key_frame)) return false; // 调用 MPP 编码

        // 构建 AVPacket 并写入 RTSP 流
        AVPacket pkt{};                 // 栈上分配的 AVPacket（避免堆分配开销）
        av_init_packet(&pkt);           // 初始化数据包（设置默认值）
        pkt.data = encoded.data();      // 数据指针（指向 encoded 内部缓冲区）
        pkt.size = static_cast<int>(encoded.size()); // 数据大小
        pkt.stream_index = 0;           // 流索引（仅一路视频，固定为 0）
        pkt.pts = pts;                  // 显示时间戳
        pkt.dts = pts;                  // 解码时间戳（无 B 帧时 PTS == DTS）
        pkt.duration = 1;               // 帧持续时间（以时间基为单位）
        if (key_frame) pkt.flags |= AV_PKT_FLAG_KEY;  // 标记关键帧
        return av_interleaved_write_frame(fmt_ctx, &pkt) >= 0; // 写入 RTSP 网络栈
    }

    // ---- FFmpeg 编码器路径（通用路径） ----
    AVCodecContext* codec_ctx = static_cast<AVCodecContext*>(avcodec_context_rtsp_); // 编码器上下文
    AVStream* stream = fmt_ctx->streams[0]; // RTSP 流中通常只有一路视频流

    // === 步骤 3：AVFrame 资源管理 ===
    // 复用帧对象以减少频繁分配内存的开销
    AVFrame* av_frame = static_cast<AVFrame*>(av_frame_cache_);
    // 若缓存为空或视频尺寸发生变化（如动态调分辨率），则重新分配帧缓冲区
    if (!av_frame || av_frame->width != codec_ctx->width || av_frame->height != codec_ctx->height) {
        if (av_frame) {
            AVFrame* tmp_frame = av_frame;
            av_frame_free(&tmp_frame);      // 释放旧的帧对象
        }
        av_frame = av_frame_alloc();        // 分配新的帧对象
        if (!av_frame) return false;        // 分配失败

        // 设置帧属性：必须与编码器配置（NV12 + 目标尺寸）完全一致
        av_frame->format = codec_ctx->pix_fmt;  // 像素格式 NV12
        av_frame->width = codec_ctx->width;     // 帧宽度
        av_frame->height = codec_ctx->height;   // 帧高度

        // 按 32 字节对齐分配帧缓冲区，满足 Rockchip 硬件编码器 DMA 对齐要求。
        if (av_frame_get_buffer(av_frame, 32) < 0) {  // align=32 对齐
            av_frame_free(&av_frame);         // 分配失败，释放帧
            return false;
        }
        av_frame_cache_ = av_frame;           // 更新类成员缓存
    } else {
        // 确保帧缓冲区可写（如果该帧被其他地方引用，会自动执行引用计数保护下的拷贝）
        if (av_frame_make_writable(av_frame) < 0) {  // 引用计数 > 1 时拷贝
            return false;
        }
    }

    // === 步骤 4：颜色空间转换（BGR → NV12） ===
    // 优先使用 RGA 2D 硬件完成 BGR -> NV12，降低 CPU 占用。
    bool rga_converted = false;                    // RGA 转换是否成功标志
    {
        // 线程局部静态变量：每个推流线程拥有独立的 RGA 处理器实例，避免多管线竞争
        static thread_local pipeline::RgaPreprocessor rga_handler;  // RGA 处理器
        static thread_local bool rga_ready = false;                 // RGA 就绪标志
        static thread_local int cache_w = 0, cache_h = 0;           // 缓存的宽高
        static thread_local cv::Mat nv12_tmp_cache;                 // NV12 临时缓冲区

        const int target_w = codec_ctx->width;      // 目标宽度
        const int target_h = codec_ctx->height;     // 目标高度

        // 动态检查：参数变化时重新初始化 RGA 配置
        if (!rga_ready || cache_w != target_w || cache_h != target_h) {
            pipeline::RgaPreprocessConfig rga_cfg;
            rga_cfg.use_rga = true;                                   // 启用 RGA
            rga_cfg.src_format = pipeline::RgaPixelFormat::BGR888;    // 源格式：OpenCV BGR
            rga_cfg.dst_format = pipeline::RgaPixelFormat::NV12;      // 目标格式：NV12（硬件编码器输入）
            rga_cfg.target_width = target_w;                          // 目标宽度
            rga_cfg.target_height = target_h;                         // 目标高度
            rga_ready = rga_handler.initialize(rga_cfg);               // 初始化 RGA
            cache_w = target_w;                                        // 缓存宽度
            cache_h = target_h;                                        // 缓存高度
        }

        // 执行硬件转换：BGR -> NV12
        if (rga_ready && rga_handler.isRgaActive()) {     // RGA 可用
            const int y_stride = av_frame->linesize[0];    // Y 平面行跨度
            const int uv_stride = av_frame->linesize[1];   // UV 平面行跨度
            const uint8_t* expected_uv = av_frame->data[0] + static_cast<ptrdiff_t>(y_stride) * target_h; // UV 起始地址
            // 公共保护条件：
            // 1. 输入必须仍是所有管线统一输出的 BGR888(OpenCV CV_8UC3)
            // 2. 编码器/AVFrame 必须是 NV12
            // 3. FFmpeg 的 Y/UV 平面布局必须连续且 stride 一致
            // 任一条件不满足，都回退到旧的 nv12_tmp_cache + memcpy 路径，避免误伤其它管线。
            const bool can_direct_write = (frame.type() == CV_8UC3) &&               // 源格式校验
                                          (codec_ctx->pix_fmt == AV_PIX_FMT_NV12) && // 编码器格式校验
                                          (av_frame->format == AV_PIX_FMT_NV12) &&   // 帧格式校验
                                          (av_frame->data[0] != nullptr) &&           // Y 平面有效
                                          (av_frame->data[1] != nullptr) &&           // UV 平面有效
                                          (av_frame->data[1] == expected_uv) &&       // UV 紧接在 Y 后（连续布局）
                                          (y_stride > 0) &&                           // stride 合法
                                          (uv_stride == y_stride);                    // Y/UV stride 一致

            // 最优路径：RGA 直接把 NV12 写进 AVFrame 背后的编码缓冲，
            // 这样就省掉了"中间 NV12 临时 Mat -> memcpy 到 AVFrame"这一步。
            if (can_direct_write && rga_handler.processToBuffer(frame, av_frame->data[0], y_stride)) {
                rga_converted = true;  // 零拷贝路径成功
            } else if (rga_handler.process(frame, nv12_tmp_cache) && !nv12_tmp_cache.empty()) {
                // 回退：先写入连续 NV12 缓冲，再拷贝到 AVFrame 的 Y/UV 平面
                const int y_size = target_w * target_h;              // Y 平面字节数
                const int uv_size = target_w * (target_h / 2);       // UV 平面字节数

                if (y_stride == target_w) {                           // stride 等于宽度（无填充）
                    memcpy(av_frame->data[0], nv12_tmp_cache.data, y_size);            // 拷贝 Y
                    memcpy(av_frame->data[1], nv12_tmp_cache.data + y_size, uv_size);  // 拷贝 UV
                } else {
                    // stride 大于宽度时逐行拷贝（处理 DMA 对齐填充）
                    for (int i = 0; i < target_h; ++i)
                        memcpy(av_frame->data[0] + i * y_stride, nv12_tmp_cache.data + i * target_w, target_w);
                    for (int i = 0; i < target_h / 2; ++i)
                        memcpy(av_frame->data[1] + i * uv_stride, nv12_tmp_cache.data + y_size + i * target_w, target_w);
                }
                rga_converted = true;  // 回退路径也标记成功
            }
        }
    }

    // 兜底路径：若 RGA 硬件故障或不可用，回退到 FFmpeg Swscale (CPU 消耗大)
    if (!rga_converted) {
        SwsContext* sws_ctx = static_cast<SwsContext*>(sws_context_cache_);  // 获取缓存
        if (!sws_ctx) {  // 首次创建 SwsContext
            sws_ctx = sws_getContext(
                frame.cols, frame.rows, AV_PIX_FMT_BGR24,          // 源：BGR 24bit
                codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt, // 目标：编码器格式（通常是 NV12）
                SWS_BILINEAR, nullptr, nullptr, nullptr);           // 双线性插值
            sws_context_cache_ = sws_ctx;                           // 缓存上下文
        }
        if (sws_ctx) {
            uint8_t* src_slices[4] = {frame.data, nullptr, nullptr, nullptr}; // 源数据切片
            int src_strides[4] = {(int)frame.step, 0, 0, 0};                // 源行跨度
            sws_scale(sws_ctx, src_slices, src_strides, 0, frame.rows,
                      av_frame->data, av_frame->linesize);                   // CPU 格式转换
        } else {
            return false; // 转换完全失败（无 SwsContext 且无 RGA），跳过该帧
        }
    }

    // === 步骤 5：送入编码器 ===
    // 设置显示时间戳 (PTS)，rtsp_frame_index_ 随帧递增确保时序正确
    av_frame->pts = rtsp_frame_index_++;                                // 单调递增 PTS
    int ret = avcodec_send_frame(codec_ctx, av_frame);                  // 将帧送入编码器
    if (ret < 0) return false;                                          // 编码器拒绝该帧

    // === 步骤 6：提取编码后的 H.264 数据包并写入 RTSP 传输链路 ===
    bool sent_ok = false;                                               // 发送结果标志
    AVPacket* pkt = static_cast<AVPacket*>(av_packet_cache_);           // 获取缓存的包对象
    if (!pkt) {                                                         // 首次分配
        pkt = av_packet_alloc();                                        // 分配新包对象
        if (!pkt) {
            return false;                                               // 分配失败
        }
        av_packet_cache_ = pkt;                                         // 缓存
    }
    // 编码器可能在接收一帧后输出零个或多个 AVPacket（取决于编码负载和帧类型：
    // I 帧可能产生多个 slice NALU，P 帧通常一个 NALU）
    while (avcodec_receive_packet(codec_ctx, pkt) >= 0) {               // 循环接收编码数据
        pkt->stream_index = stream->index;                              // 设置流索引
        // 关键时序转换：将物理帧的时间基（1/FPS）映射到网络传输流的时间基（通常为 90k 采样率）
        av_packet_rescale_ts(pkt, codec_ctx->time_base, stream->time_base);

        // 真正的网络发送动作：写入 RTSP 网络栈
        if (av_interleaved_write_frame(fmt_ctx, pkt) >= 0) {           // 交错写入保证时序
            sent_ok = true;                                             // 标记发送成功
        }
        av_packet_unref(pkt); // 释放 Packet 内部数据引用，不释放 Packet 对象本身（用于下一轮循环复用）
    }

    return sent_ok;  // 返回发送结果（至少一个包成功即视为发送成功）
}

/**
 * @brief 获取推流统计信息
 * @return 包含发送帧数、丢帧数、实时 FPS 等统计数据的快照
 */
StreamingManager::StreamingStats StreamingManager::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);  // 加锁保证统计信息线程安全
    return stats_;                                    // 返回统计结构体副本
}

/**
 * @brief RTSP 自动重连
 *
 * 当连续发送失败达到阈值时触发，执行以下恢复流程：
 *   1. 释放旧连接的 FFmpeg 格式上下文和编码器上下文
 *   2. 释放 MPP 编码器
 *   3. 清空所有缓存（SwsContext、AVFrame、AVPacket）
 *   4. 重置帧序号（避免 PTS 不连续）
 *   5. 重新初始化 RTSP 输出
 *
 * @return 重连成功返回 true，失败返回 false
 * @note 仅在工作线程（streamingWorker）中调用，无需额外加锁
 */
bool StreamingManager::tryReconnectRTSP() {
    // 释放旧连接资源
    closeOutput(reinterpret_cast<AVFormatContext**>(&avformat_context_rtsp_),
                reinterpret_cast<AVCodecContext**>(&avcodec_context_rtsp_));  // 释放 FFmpeg 上下文
    if (mpp_encoder_) {
        delete static_cast<MppH264Encoder*>(mpp_encoder_);  // 释放 MPP 编码器
        mpp_encoder_ = nullptr;                              // 置空
    }

    // 释放缓存的转换上下文与帧（避免尺寸/格式不匹配导致错误）
    if (sws_context_cache_) {
        sws_freeContext(static_cast<SwsContext*>(sws_context_cache_));  // 释放 SwsContext
        sws_context_cache_ = nullptr;
    }
    if (av_frame_cache_) {
        AVFrame* cached = static_cast<AVFrame*>(av_frame_cache_);
        av_frame_free(&cached);         // 释放 AVFrame 及其内部缓冲区
        av_frame_cache_ = nullptr;
    }
    if (av_packet_cache_) {
        AVPacket* cached_pkt = static_cast<AVPacket*>(av_packet_cache_);
        av_packet_free(&cached_pkt);    // 释放 AVPacket 对象
        av_packet_cache_ = nullptr;
    }

    // 重置帧序号（避免 PTS 从之前的值继续递增导致异常）
    rtsp_frame_index_ = 0;

    // 重新初始化 RTSP（走完整的编码器探测流程）
    if (!initializeRTSP()) {
        return false;  // 重连失败
    }
    return true;       // 重连成功
}
