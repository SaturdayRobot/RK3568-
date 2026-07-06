/**
 * @file stream_loader.cpp
 * @brief 视频流加载与管理模块实现
 *
 * 该文件实现了StreamLoader类的核心功能，
 * 包括视频流的打开、读取、解码、帧回调处理以及流的管理。
 *
 * 数据流向：
 *   网络/文件 → FFmpeg(av_read_frame) → BSF(H.264格式转换) → MPP Decoder(硬解)
 *   → mpp_decoder_frame_callback(零拷贝DMA-BUF) → onDecodedFrame(入队)
 *   → waitAndGetDecodedFrame(出队) → RGA/NPU(后续处理)
 */

#include "data_collection/stream_loader.h"

#include <algorithm>
#include <iostream>

// ============================================================================
// Annex B 格式检测
// ============================================================================

/**
 * @brief 判断数据缓冲区是否为 Annex B 格式
 *
 * Annex B 是 H.264/H.265 字节流格式，每个NAL单元以起始码开头：
 * - 3字节: 0x000001
 * - 4字节: 0x00000001
 *
 * MPP硬件解码器要求输入必须为Annex B格式。
 * 另一种常见格式是AVCC（长度前缀），常见于MP4容器，需通过BSF转换。
 */
int is_annexb(const uint8_t *buf, size_t buf_size)
{
    if (buf_size >= 4) {
        if ((buf[0] == 0x00 && buf[1] == 0x00 && buf[2] == 0x01) ||
            (buf[0] == 0x00 && buf[1] == 0x00 && buf[2] == 0x00 && buf[3] == 0x01)) {
            return 1;
        }
    }
    return 0;
}

// ============================================================================
// MPP 解码回调 — C风格回调，桥接MPP SDK与StreamLoader
// ============================================================================

/**
 * @brief MPP解码器帧回调函数
 *
 * MPP解码器每完成一帧NV12硬件解码即回调此函数。
 *
 * 关键设计 — 全链路零拷贝：
 * - 仅透传DMA-BUF FD与帧生命周期令牌(frame_hold_token)
 * - 不做像素格式转换或内存拷贝
 * - BGR物化延迟到下游按需执行（通常由RGA硬件完成）
 *
 * 格式安全校验：
 * - 仅接受线性NV12 (YUV420SP) 格式
 * - AFBC压缩/tile排布/10-bit色深等变体会导致偏色或驱动错误，直接丢弃
 */
void mpp_decoder_frame_callback(void *buffer,
                                int width_stride,
                                int height_stride,
                                int width,
                                int height,
                                int format,
                                int fd,
                                void *data,
                                size_t buffer_size,
                                int id,
                                const std::shared_ptr<void>& frame_hold_token)
{
    StreamLoader* loader = reinterpret_cast<StreamLoader*>(buffer);
    if (!loader) return;

    (void)data;
    (void)id;

    // 格式安全校验：仅接受线性NV12
    const int base_format = format & MPP_FRAME_FMT_MASK;
    const bool linear = (format & (MPP_FRAME_FBC_MASK | MPP_FRAME_TILE_FLAG)) == 0;
    if (!linear || base_format != MPP_FMT_YUV420SP) {
        std::cerr << "[stream_loader] unsupported MPP frame format=0x"
                  << std::hex << format << std::dec << ", drop frame" << std::endl;
        return;
    }

    // 帧参数有效性校验
    if (fd <= 0 || width <= 0 || height <= 0 ||
        width_stride < width || height_stride < height || buffer_size == 0) {
        std::cerr << "[stream_loader] invalid dma fd from MPP, drop frame" << std::endl;
        return;
    }

    // 构建硬件帧描述符（零拷贝：仅透传FD与生命周期令牌）
    StreamLoader::DecodedHwFrameDesc hw_desc;
    hw_desc.dma_fd = fd;
    hw_desc.width = width;
    hw_desc.height = height;
    hw_desc.width_stride = width_stride;
    hw_desc.height_stride = height_stride;
    hw_desc.buffer_size = buffer_size;
    hw_desc.frame_hold = frame_hold_token;

    loader->onDecodedFrame(hw_desc);
}

// ============================================================================
// 帧消费者接口 — waitAndGetDecodedFrame / releaseDecodedFrame
// ============================================================================

bool StreamLoader::waitAndGetDecodedFrame(uint64_t& out_desc_id,
                                          DecodedHwFrameDesc& out_hw_desc,
                                          std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(desc_mutex_);

    // 消费者阻塞点：condition_variable::wait_for 等待生产者通知
    // 唤醒条件：(1) ready_queue_ 非空 → 有帧可消费  (2) stopFlag → 流停止
    if (!desc_cv_.wait_for(lock, timeout, [this]() {
            return !ready_queue_.empty() || stopFlag.load();
        })) {
        return false; // 超时
    }

    if (ready_queue_.empty()) {
        return false; // stopFlag 触发的唤醒，无帧可消费
    }

    // FIFO 出队
    const uint64_t desc_id = ready_queue_.front();
    ready_queue_.pop_front();

    // 校验帧槽位有效性
    auto it = out_desc_to_holder_.find(desc_id);
    if (it == out_desc_to_holder_.end() || !it->second || !it->second->hw_desc.valid()) {
        return false;
    }

    out_hw_desc = it->second->hw_desc;
    out_desc_id = desc_id;
    return true;
}

void StreamLoader::releaseDecodedFrame(uint64_t desc_id)
{
    if (desc_id == 0) return;

    std::lock_guard<std::mutex> lock(desc_mutex_);
    // 延迟回收：推入回收队列，立即排空（避免在热路径上直接操作map）
    recycle_queue_.push_back(desc_id);
    drainPendingRecycleQueueLocked();
}

// ============================================================================
// 帧生产者接口 — onDecodedFrame（回调→入队核心逻辑）
// ============================================================================

void StreamLoader::onDecodedFrame(const DecodedHwFrameDesc& hw_desc)
{
    if (!hw_desc.valid()) return;

    std::lock_guard<std::mutex> lock(desc_mutex_);

    // 步骤1: 排空回收队列，释放已消费holder
    drainPendingRecycleQueueLocked();

    // 步骤2: Holder Map 上限保护（防止描述符ID泄漏导致内存无限增长）
    // 只能淘汰尚未被消费者取走的帧（仍在 ready_queue_ 中的帧），
    // 已出队但未回收的帧可能正被RGA/NPU读取，不可擅自释放。
    constexpr size_t kMaxHolderMapSize = 64;
    while (out_desc_to_holder_.size() >= kMaxHolderMapSize && !ready_queue_.empty()) {
        const uint64_t oldest_ready_id = ready_queue_.front();
        ready_queue_.pop_front();
        out_desc_to_holder_.erase(oldest_ready_id);
        dropped_holder_count_++;
    }

    // 步骤3: 就绪队列深度控制 — "保最新、丢最旧"策略
    // RTSP实时链路优先低延迟，积压帧仅增加端到端延迟，丢弃最旧帧更合理。
    while (ready_queue_.size() >= max_ready_depth_) {
        const uint64_t old_id = ready_queue_.front();
        ready_queue_.pop_front();
        out_desc_to_holder_.erase(old_id);
        dropped_ready_count_++;
    }

    // 步骤4: 创建帧槽位 → 分配desc_id → 入队 → 通知消费者
    auto holder = std::make_shared<DecodedFrameSlot>();
    holder->hw_desc = hw_desc;

    const uint64_t desc_id = next_desc_id_++;
    out_desc_to_holder_[desc_id] = std::move(holder);
    ready_queue_.push_back(desc_id);
    desc_cv_.notify_one();
}

// ============================================================================
// 运行时配置与统计
// ============================================================================

void StreamLoader::configureRuntime(const RuntimeOptions& options) {
    std::lock_guard<std::mutex> lock(desc_mutex_);
    max_ready_depth_ = std::max<size_t>(1, options.max_ready_depth);
    rtbufsize_bytes_ = std::max(1024, options.rtbufsize_bytes);
    stimeout_us_ = std::max(100000, options.stimeout_us);
    max_delay_us_ = std::max(0, options.max_delay_us);
    rtsp_transport_ = options.rtsp_transport.empty() ? "tcp" : options.rtsp_transport;
}

size_t StreamLoader::readyDepth() const {
    std::lock_guard<std::mutex> lock(desc_mutex_);
    return ready_queue_.size();
}

size_t StreamLoader::maxReadyDepth() const {
    std::lock_guard<std::mutex> lock(desc_mutex_);
    return max_ready_depth_;
}

std::uint64_t StreamLoader::droppedReadyCount() const {
    std::lock_guard<std::mutex> lock(desc_mutex_);
    return dropped_ready_count_;
}

std::uint64_t StreamLoader::droppedHolderCount() const {
    std::lock_guard<std::mutex> lock(desc_mutex_);
    return dropped_holder_count_;
}

std::uint64_t StreamLoader::droppedTotalCount() const {
    std::lock_guard<std::mutex> lock(desc_mutex_);
    return dropped_ready_count_ + dropped_holder_count_;
}

void StreamLoader::drainPendingRecycleQueueLocked()
{
    // 调用前必须已持有 desc_mutex_（函数名 Locked 后缀表示此约定）
    while (!recycle_queue_.empty()) {
        const uint64_t id = recycle_queue_.front();
        recycle_queue_.pop_front();
        out_desc_to_holder_.erase(id);
    }
}

// ============================================================================
// 流生命周期管理 — close / open / operator()
// ============================================================================

/**
 * @brief 关闭流并释放所有资源
 *
 * 释放顺序：MPP解码器 → BSF上下文 → AVDictionary → AVPacket →
 * AVFormatContext → AVCodecParameters → 帧描述符队列。
 * 每步均有空指针检查，方法幂等，可安全多次调用。
 */
void StreamLoader::close()
{
    decoder.Reset();

    if (bsf_ctx) {
        av_bsf_free(&bsf_ctx);
        bsf_ctx = nullptr;
    }

    if (options) {
        av_dict_free(&options);
        options = nullptr;
    }

    if (temp_pkt) {
        av_packet_free(&temp_pkt);
        temp_pkt = nullptr;
    }

    if (fmtCtx) {
        // avformat_close_input: TEARDOWN网络连接 + 释放所有内部streams + 释放上下文
        avformat_close_input(&fmtCtx);
        fmtCtx = nullptr;
    }

    if (codecPar) {
        avcodec_parameters_free(&codecPar);
        codecPar = nullptr;
    }

    videoStreamIndex = -1;
    width = 0;
    height = 0;
    isnotAnnexB = false;

    // 清理帧队列并唤醒所有等待者
    {
        std::lock_guard<std::mutex> lock(desc_mutex_);
        ready_queue_.clear();
        recycle_queue_.clear();
        out_desc_to_holder_.clear();
        desc_cv_.notify_all();
    }
}

/**
 * @brief 从视频流中读取一帧数据并提交硬件解码
 *
 * 单帧处理流水线：
 * 1. av_read_frame → 从网络/文件读取原始码流包
 * 2. 过滤非视频包 → 丢弃音频等其他类型
 * 3. BSF格式转换 → H.264的MP4→AnnexB转换（如需要）
 * 4. MPP硬件解码 → 提交给Rockchip VPU（解码结果通过回调异步返回）
 *
 * 容错：最多重试10次，每次失败后睡眠2ms。
 */
bool StreamLoader::read_frame()
{
    using namespace std::chrono_literals;
    int retry_times = 0;

    do {
        int x = av_read_frame(fmtCtx, temp_pkt);

        if (x < 0) {
            std::cerr << "av_read_frame 失败 " << retry_times << std::endl;
            status = x;
            std::this_thread::sleep_for(2ms);
            av_packet_unref(temp_pkt);
            continue;
        }

        if (temp_pkt->stream_index == videoStreamIndex) {
            // BSF格式转换：H.264的MP4/AVCC→AnnexB（MPP硬解要求AnnexB输入）
            if (isnotAnnexB) {
                int ret = av_bsf_send_packet(bsf_ctx, temp_pkt);
                if (ret < 0) {
                    fprintf(stderr, "Error sending packet to filter\n");
                    break;
                }
                ret = av_bsf_receive_packet(bsf_ctx, temp_pkt);
                if (ret < 0) {
                    fprintf(stderr, "Error receiving packet from filter\n");
                    break;
                }
            }

            // 提交给MPP硬件解码器（异步：解码结果通过回调返回）
            bool decode_success = decoder.Decode(temp_pkt->data, temp_pkt->size, 0);
            av_packet_unref(temp_pkt);

            if (decode_success) {
                status = 0;
                return true;
            } else {
                std::this_thread::sleep_for(2ms);
            }
        } else {
            // 非视频包直接丢弃
            av_packet_unref(temp_pkt);
            continue;
        }
    } while (++retry_times < 10);

    return false;
}

/**
 * @brief 构造函数 — 初始化流加载器核心状态
 */
StreamLoader::StreamLoader(char *url, int id)
{
    stream_loader_id = id;
    std::cout << "StreamLoader: " << std::to_string(id) << std::endl;

    // 绑定C风格回调 → MPP解码完成后回调 mpp_decoder_frame_callback
    // → 内部将userdata转型回StreamLoader → 调用onDecodedFrame入队
    callback = mpp_decoder_frame_callback;

    stream_url = url;
    status = 0;
    stopFlag = false;
}

/**
 * @brief 析构函数 — 自动释放所有资源
 */
StreamLoader::~StreamLoader()
{
    std::cout << "destory stream loader: " << stream_loader_id << std::endl;
    close();
}

/**
 * @brief 打开并初始化视频流
 *
 * 一次性串联"网络拉流 + 码流适配 + MPP硬解 + 回调绑定"的完整初始化流程。
 * 成功后 operator() 线程只需持续 read_frame() 即可。
 *
 * 初始化步骤：
 * 1. 防御式close()清理残留
 * 2. 分配AVPacket/AVCodecParameters
 * 3. 配置FFmpeg网络选项（低延迟、TCP重连、实时模式）
 * 4. avformat_open_input（RTSP: OPTIONS→DESCRIBE→SETUP→PLAY）
 * 5. avformat_find_stream_info（解析SPS/PPS获取编码参数和分辨率）
 * 6. 遍历流定位视频流索引
 * 7. 根据编码格式(H.264/H.265)初始化MPP硬解管道
 *    - H.264: 额外配置 h264_mp4toannexb BSF过滤器
 *    - H.265: 原生AnnexB格式，无需BSF
 * 8. 绑定解码回调 + 拷贝编解码参数
 *
 * @return 0=成功, -1=FFmpeg失败, -2=无视频流, -3=解码器失败,
 *         -4=不支持编码格式, -5=参数复制失败
 */
int StreamLoader::open()
{
    // 防御式清理
    close();

    // 分配临时数据包和编解码参数
    temp_pkt = av_packet_alloc();
    codecPar = avcodec_parameters_alloc();
    if (!temp_pkt || !codecPar) {
        std::cerr << "alloc temp_pkt/codecPar failed" << std::endl;
        close();
        return -1;
    }

    // 配置FFmpeg网络选项 — 优化RTSP实时流的低延迟和可靠性
    const std::string rtbufsize_str = std::to_string(rtbufsize_bytes_);
    const std::string stimeout_str = std::to_string(stimeout_us_);
    const std::string max_delay_str = std::to_string(max_delay_us_);

    av_dict_set(&options, "rtbufsize", rtbufsize_str.c_str(), 0);
    av_dict_set(&options, "start_time_realtime", 0, 0);
    av_dict_set(&options, "rtsp_transport", rtsp_transport_.c_str(), 0);
    av_dict_set(&options, "stimeout", stimeout_str.c_str(), 0);
    av_dict_set(&options, "max_delay", max_delay_str.c_str(), 0);
    av_dict_set(&options, "buffer_size", "524288", 0);
    av_dict_set(&options, "fflags", "nobuffer", 0);       // 实时流不做额外探测缓存
    av_dict_set(&options, "flags", "low_delay", 0);        // 低延迟模式
    av_dict_set(&options, "reorder_queue_size", "0", 0);   // 关闭B帧重排序队列
    av_dict_set(&options, "flush_packets", "1", 0);        // 每包立即刷新
    av_dict_set(&options, "reconnect", "1", 0);            // 启用TCP自动重连
    av_dict_set(&options, "reconnect_delay_max", "5", 0);  // 最大重连间隔5秒

    // 打开输入流（RTSP协议协商）
    if (avformat_open_input(&fmtCtx, stream_url, NULL, &options) != 0) {
        std::cout << "open rtsp stream failed" << std::endl;
        close();
        return -1;
    }

    // 探测流信息（读取数据包解析编码格式、分辨率等）
    if (avformat_find_stream_info(fmtCtx, NULL) < 0) {
        close();
        return -1;
    }

    av_dump_format(fmtCtx, 0, stream_url, 0);

    // 定位第一个视频流
    videoStreamIndex = -1;
    for (unsigned int i = 0; i < fmtCtx->nb_streams; i++) {
        if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            width = fmtCtx->streams[i]->codecpar->width;
            height = fmtCtx->streams[i]->codecpar->height;
            videoStreamIndex = i;
            break;
        }
    }

    std::cout << "videoindex: " << videoStreamIndex << std::endl;
    if (videoStreamIndex < 0) {
        close();
        return -2;
    }

    // 初始化MPP硬件解码器
    AVCodecID rtsp_format = fmtCtx->streams[videoStreamIndex]->codecpar->codec_id;

    if (status == 0) {
        int ret = 0;
        void *src_buffer = this;

        switch (rtsp_format) {
        case AV_CODEC_ID_H264:
            ret = decoder.Init(264, 25, src_buffer, stream_loader_id);
            if (ret <= 0) {
                std::cerr << "H264 decoder init failed" << std::endl;
                close();
                return -3;
            }

            // H.264特有：配置 h264_mp4toannexb 比特流过滤器
            // MPP要求AnnexB格式输入，而MP4容器使用AVCC（长度前缀）格式
            bsf = av_bsf_get_by_name("h264_mp4toannexb");
            if (!bsf) {
                fprintf(stderr, "Could not find h264_mp4toannexb filter\n");
                close();
                return -3;
            }
            if (av_bsf_alloc(bsf, &bsf_ctx) < 0) {
                fprintf(stderr, "Could not allocate bsf context\n");
                close();
                return -3;
            }
            avcodec_parameters_copy(bsf_ctx->par_in,
                                    fmtCtx->streams[videoStreamIndex]->codecpar);
            bsf_ctx->time_base_in = fmtCtx->streams[videoStreamIndex]->time_base;
            if (av_bsf_init(bsf_ctx) < 0) {
                fprintf(stderr, "Could not initialize bsf context\n");
                av_bsf_free(&bsf_ctx);
                close();
                return -3;
            }
            isnotAnnexB = true;
            std::cout << "H264 " << ret << std::endl;
            break;

        case AV_CODEC_ID_HEVC:
            // H.265原生使用AnnexB格式，无需BSF转换
            ret = decoder.Init(265, 25, src_buffer, stream_loader_id);
            if (ret <= 0) {
                std::cerr << "HEVC decoder init failed" << std::endl;
                close();
                return -3;
            }
            std::cout << "HEVC " << ret << std::endl;
            break;

        default:
            std::cerr << "Unsupported codec id: "
                      << static_cast<int>(rtsp_format) << std::endl;
            close();
            return -4;
        }
    }

    // 绑定解码回调（零拷贝DMA-BUF传递入口）
    decoder.SetCallback(this->callback);

    // 保存编解码参数供外部查询
    if (avcodec_parameters_copy(codecPar,
                                fmtCtx->streams[videoStreamIndex]->codecpar) < 0) {
        std::cerr << "avcodec_parameters_copy failed" << std::endl;
        close();
        return -5;
    }

    return 0;
}

/**
 * @brief 线程入口函数（operator() 重载）
 *
 * 独立线程主循环。调用方式：std::thread t(std::ref(loader));
 *
 * 线程模型：
 * - 单生产者线程持续 av_read_frame → MPP解码
 * - MPP回调异步推送帧到 ready_queue_（生产者-消费者队列）
 * - 上层管线通过 waitAndGetDecodedFrame 阻塞消费
 *
 * 自动重连（指数退避）：
 * - status != 0 时触发：close() → open() 循环
 * - 退避序列: 1s → 2s → 4s → 8s → 16s → 30s(上限)
 * - 退避期间每100ms检查 stopFlag，保证快速响应停止信号
 * - 退出时 notify_all() 唤醒所有阻塞消费者
 */
void StreamLoader::operator()()
{
    while (stopFlag == false) {
        try {
            if (!read_frame()) {
                std::cout << "read frame error " << stream_loader_id << std::endl;
            }
        } catch (std::exception &e) {
            std::cout << "exception ............" << std::endl;
            std::cout << e.what() << std::endl;
        }

        // 流错误检测与自动重连 — RTSP采集链路自愈核心
        if (status) {
            std::cout << "\nstatus: " << status << std::endl;
            std::cout << "Reconnecting " << stream_loader_id << std::endl;

            close();

            // 指数退避重连循环
            int backoff_ms = 1000;
            const int max_backoff_ms = 30000;
            while (open() != 0 && !stopFlag) {
                std::cout << "Reconnect failed, retry in " << backoff_ms
                          << "ms (stream " << stream_loader_id << ")" << std::endl;

                // 分段睡眠，每100ms检查stopFlag，避免长时间阻塞无法退出
                auto deadline = std::chrono::steady_clock::now()
                                + std::chrono::milliseconds(backoff_ms);
                while (std::chrono::steady_clock::now() < deadline && !stopFlag) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                if (stopFlag) break;

                backoff_ms = std::min(backoff_ms * 2, max_backoff_ms);
            }

            if (stopFlag) break;
            status = 0; // 重连成功，恢复正常状态
        }
    }

    // 优雅退出：唤醒所有等待消费者
    desc_cv_.notify_all();
}
