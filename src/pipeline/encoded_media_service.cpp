#include "pipeline/encoded_media_service.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/mem.h>
}

#include "pipeline/rga_preprocessor.h"
#include "utils/thread_runtime.h"

namespace pipeline {
namespace {

// RtspPacketWriter: 基于 FFmpeg libavformat 的 RTSP 推流写入器
// - 负责建立 RTSP 连接、写入 H.264 码流头部（SPS/PPS）、逐包推送编码数据
// - 支持断线重连：重连后重新发送 SPS/PPS 头部，从下一个 IDR 关键帧开始推送
class RtspPacketWriter {
public:
    ~RtspPacketWriter() { close(); }

    // open(): 建立 RTSP 输出连接并写入 H.264 码流头部（SPS/PPS extradata）
    // - config: 编码媒体配置（含 RTSP URL、分辨率）
    // - header: H.264 extradata（SPS + PPS），在重连时必须重新发送
    // - 返回值: true 表示连接建立成功
    bool open(const EncodedMediaConfig& config, const std::vector<uint8_t>& header) {
        close();                                                             // 先关闭已有连接
        avformat_network_init();                                             // 初始化 FFmpeg 网络模块
        // 分配 RTSP 输出上下文
        if (avformat_alloc_output_context2(&context_, nullptr, "rtsp", config.rtsp_url.c_str()) < 0 ||
            !context_) return false;
        // 创建视频流
        AVStream* stream = avformat_new_stream(context_, nullptr);
        if (!stream) return false;
        stream->time_base = AVRational{1, 90000};                            // RTSP 时间基 1/90000（H.264 标准时钟）
        stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;                   // 流类型：视频
        stream->codecpar->codec_id = AV_CODEC_ID_H264;                       // 编码格式：H.264
        stream->codecpar->width = config.width;                              // 视频宽度
        stream->codecpar->height = config.height;                            // 视频高度
        // 分配并拷贝 extradata（SPS/PPS），附加 AV_INPUT_BUFFER_PADDING_SIZE 填充字节
        stream->codecpar->extradata = static_cast<uint8_t*>(
            av_mallocz(header.size() + AV_INPUT_BUFFER_PADDING_SIZE));
        if (!stream->codecpar->extradata) return false;
        std::memcpy(stream->codecpar->extradata, header.data(), header.size());
        stream->codecpar->extradata_size = static_cast<int>(header.size());

        // 配置 RTSP 输出选项（通过 AVDictionary 传递）
        AVDictionary* options = nullptr;
        av_dict_set(&options, "rtsp_transport", "tcp", 0);    // 使用 TCP 传输（可靠传输，穿透防火墙）
        av_dict_set(&options, "muxdelay", "0", 0);             // 零复用延迟（实时推流不需要缓冲）
        av_dict_set(&options, "flush_packets", "1", 0);        // 每包立即刷新（降低延迟）
        av_dict_set(&options, "avioflags", "direct", 0);       // 直接 I/O，减少缓冲
        av_dict_set(&options, "tcp_nodelay", "1", 0);          // TCP_NODELAY 选项（禁用 Nagle 算法）
        av_dict_set(&options, "buffer_size", "262144", 0);     // 发送缓冲区 256KB
        av_dict_set(&options, "stimeout", "2000000", 0);       // 发送超时 2 秒（微秒）
        av_dict_set(&options, "rw_timeout", "2000000", 0);     // 读写超时 2 秒（微秒）
        int result = 0;
        // 如果输出格式需要显式打开 I/O（RTSP 需要），则调用 avio_open2
        if (!(context_->oformat->flags & AVFMT_NOFILE)) {
            result = avio_open2(&context_->pb, config.rtsp_url.c_str(), AVIO_FLAG_WRITE,
                                nullptr, &options);
        }
        // 写入容器头部（包含 SPS/PPS extradata），客户端由此初始化解码器
        if (result >= 0) result = avformat_write_header(context_, &options);
        av_dict_free(&options);                                              // 释放字典资源
        opened_ = result >= 0;                                               // 记录打开状态
        if (!opened_) close();                                               // 失败时清理资源
        return opened_;
    }

    // write(): 将一帧 H.264 编码包写入 RTSP 流
    // - packet: 编码后的数据包，包含 PTS/DTS/时长/关键帧标志
    // - 返回值: true 表示写入成功
    // - 内部完成纳秒时间基到 RTSP 90000 时间基的转换
    bool write(const recording::EncodedPacket& packet) {
        if (!opened_ || !context_ || !packet.data || packet.data->empty()) return false;
        // 以首个包的 PTS 为基准零点，后续包计算相对 PTS
        if (base_pts_ns_ < 0) base_pts_ns_ = packet.pts_ns;
        AVPacket output{};
        av_init_packet(&output);
        output.data = packet.data->data();                                   // 指向编码数据的指针
        output.size = static_cast<int>(packet.data->size());                 // 编码数据大小
        output.stream_index = 0;                                             // 流索引（仅有一个视频流）
        // 时间基转换：从纳秒（1/1e9）转换到 RTSP 标准时间基（1/90000）
        const AVRational ns_base{1, 1000000000};                             // 纳秒时间基
        const AVRational stream_base{1, 90000};                              // RTSP 流时间基
        output.pts = av_rescale_q(packet.pts_ns - base_pts_ns_, ns_base, stream_base);  // 展示时间戳
        output.dts = av_rescale_q(packet.dts_ns - base_pts_ns_, ns_base, stream_base);  // 解码时间戳
        // 帧时长（至少 1 个时间基单位），用于客户端播放同步
        output.duration = std::max<int64_t>(1,
            av_rescale_q(packet.duration_ns, ns_base, stream_base));
        // 标记关键帧（IDR 帧），客户端从关键帧开始解码
        if (packet.key_frame) output.flags |= AV_PKT_FLAG_KEY;
        // 交错写入（对单流等同于 av_write_frame，但保持 API 兼容性）
        return av_interleaved_write_frame(context_, &output) >= 0;
    }

    // close(): 关闭 RTSP 连接，写入尾部并释放资源
    void close() {
        if (!context_) return;
        if (opened_) av_write_trailer(context_);                             // 写入容器尾部
        // 关闭 I/O 上下文（如果已打开）
        if (!(context_->oformat->flags & AVFMT_NOFILE) && context_->pb) avio_closep(&context_->pb);
        avformat_free_context(context_);                                     // 释放 FFmpeg 上下文
        context_ = nullptr;
        opened_ = false;
        base_pts_ns_ = -1;                                                   // 复位基准 PTS
    }

private:
    AVFormatContext* context_ = nullptr;   // FFmpeg 输出格式上下文
    bool opened_ = false;                  // 是否已成功打开连接
    int64_t base_pts_ns_ = -1;            // 基准 PTS（纳秒），用于计算相对时间戳
};

}  // namespace

// EncodedMediaService 构造函数
// - config: 编码媒体配置（分辨率、帧率、码率、RTSP URL、录像参数等）
// - 初始化帧队列（Latest 模式：满时丢弃旧帧保留最新帧）和 RTSP 包队列（Bounded 模式：满时阻塞）
EncodedMediaService::EncodedMediaService(EncodedMediaConfig config)
    : config_(std::move(config)),
      // 帧队列：容量至少为 1，Latest 模式保证始终保留最新帧，旧帧被覆盖
      frame_queue_(static_cast<size_t>(std::max(1, config_.frame_queue_size)), QueueMode::Latest),
      // RTSP 包队列：容量至少为 2，Bounded 模式在满时阻塞写入方
      rtsp_queue_(static_cast<size_t>(std::max(2, config_.rtsp_packet_queue_size)), QueueMode::Bounded),
      recorder_(config_.recorder) {}  // 事件录像器（管理录像文件的启停）

EncodedMediaService::~EncodedMediaService() { stop(); }

// start(): 启动编码媒体服务
// - 初始化 MPP H.264 硬件编码器
// - 获取并保存 SPS/PPS 头部
// - 启动录像器
// - 启动编码线程和 RTSP 推流线程
// - 返回值: true 表示启动成功
bool EncodedMediaService::start() {
    if (running_.load()) return true;                                        // 已在运行，幂等返回
    if (config_.rtsp_url.empty()) config_.enable_rtsp = false;              // 未配置 RTSP URL，禁用推流
    // 创建 MPP H.264 硬件编码器（Rockchip MPP 平台）
    encoder_ = std::make_unique<MppH264Encoder>();
    // 初始化编码器：分辨率、帧率、目标码率
    if (!encoder_->initialize(config_.width, config_.height, config_.fps, config_.bitrate)) {
        encoder_.reset();
        return false;
    }
    // 获取 H.264 extradata（SPS + PPS），供录像器和 RTSP 推流器使用
    codec_header_ = encoder_->header();
    // 启动录像器，传入 SPS/PPS 头部和分辨率
    if (!recorder_.start(codec_header_, config_.width, config_.height)) {
        encoder_.reset();
        return false;
    }
    // 重置队列状态（清空残留数据）
    frame_queue_.reset();
    rtsp_queue_.reset();
    running_.store(true);                                                    // 设置运行标志
    // 启动编码线程：从帧队列取帧 -> RGA 转 NV12 -> MPP 编码 -> 分发到录像和 RTSP 队列
    encoder_thread_ = std::thread(&EncodedMediaService::encoderLoop, this);
    // 仅在启用 RTSP 且 URL 非空时启动 RTSP 推流线程
    if (config_.enable_rtsp && !config_.rtsp_url.empty()) {
        rtsp_thread_ = std::thread(&EncodedMediaService::rtspLoop, this);
    }
    return true;
}

// stop(): 停止编码媒体服务，依次关闭队列、join 线程、停止录像器和编码器
void EncodedMediaService::stop() {
    if (!running_.exchange(false)) return;                                   // 原子停止，避免重复
    frame_queue_.close();                                                    // 关闭帧队列，解除 pop 阻塞
    rtsp_queue_.close();                                                     // 关闭 RTSP 队列
    if (encoder_thread_.joinable()) encoder_thread_.join();                  // 等待编码线程退出
    if (rtsp_thread_.joinable()) rtsp_thread_.join();                        // 等待 RTSP 线程退出
    recorder_.stop();                                                        // 停止录像器
    encoder_.reset();                                                        // 释放编码器
}

// submitFrame(): 上游（如 MosaicStreamPipeline）调用，提交一帧拼图成品到编码管线
// - frame: 已合成完成的画面帧（BGR888 格式）
// - 非阻塞：帧被推入 Latest 模式队列，编码线程异步消费
void EncodedMediaService::submitFrame(ComposedFrame frame) {
    if (running_.load() && !frame.image.empty()) frame_queue_.push(std::move(frame));
}

// trigger(): 触发录像事件（如移动侦测、AI 检测到目标等），转发给录像器
void EncodedMediaService::trigger(const recording::Event& event) { recorder_.trigger(event); }

// updateDetection(): 更新 AI 检测状态，转发给录像器用于事件触发判断
// - type: 事件类型（如 PersonDetected、FireDetected）
// - detected: 是否检测到目标
// - mono_ns: 单调时钟纳秒时间戳
// - real_ms: 系统时钟毫秒时间戳
// - camera_id: 摄像头标识
void EncodedMediaService::updateDetection(recording::EventType type, bool detected,
                                          int64_t mono_ns, int64_t real_ms, int camera_id) {
    recorder_.updateDetection(type, detected, mono_ns, real_ms, camera_id);
}

// setRecordingCompletionCallback(): 设置录像完成回调，供上层获知录像文件生成完毕
void EncodedMediaService::setRecordingCompletionCallback(
    recording::EventRecorder::CompletionCallback callback) {
    recorder_.setCompletionCallback(std::move(callback));
}

// encoderLoop(): 编码线程主循环
// - 从帧队列取出 BGR 图像
// - 通过 RGA 硬件转换为 NV12（H.264 编码器输入格式）
// - 调用 MPP 硬件编码器生成 H.264 码流
// - 将编码包分发到录像器和 RTSP 队列
void EncodedMediaService::encoderLoop() {
    utils::applyThreadRuntime("media_encoder", "media-encoder");  // 设置线程调度策略和名称
    // 初始化 RGA 预处理器：BGR888 -> NV12 格式转换
    RgaPreprocessor converter;
    RgaPreprocessConfig rga_config;
    rga_config.use_rga = true;                                               // 启用 RGA 硬件加速
    rga_config.strict_hardware = true;                                       // 严格要求硬件支持
    rga_config.target_width = config_.width;                                 // 目标宽度（编码分辨率）
    rga_config.target_height = config_.height;                               // 目标高度
    rga_config.src_format = RgaPixelFormat::BGR888;                          // 源格式：BGR888
    rga_config.dst_format = RgaPixelFormat::NV12;                            // 目标格式：NV12（H.264 编码器输入）
    const bool converter_ready = converter.initialize(rga_config) && converter.isRgaActive();
    int64_t encode_index = 0;                                                // 编码帧序号（传递给编码器）
    int64_t last_pts_ns = 0;                                                 // 上一帧的 PTS（纳秒）
    const int64_t default_duration = 1000000000LL / std::max(1, config_.fps); // 默认帧时长（纳秒）

    // 主循环：运行中或队列非空时持续处理
    while (running_.load() || !frame_queue_.empty()) {
        ComposedFrame frame;
        // 从帧队列取出待编码帧（200ms 超时，超时后检查 running 标志）
        if (!frame_queue_.pop(frame, std::chrono::milliseconds(200))) continue;
        // 计算端到端延迟：当前时间 - 采集时间
        const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (frame.capture_mono_ns > 0 && now_ns >= frame.capture_mono_ns) {
            last_frame_age_ms_.store((now_ns - frame.capture_mono_ns) / 1000000LL); // 记录帧年龄（毫秒）
        }
        // 跳过无效帧或非 BGR888 格式的帧，以及 RGA 不可用或转换失败的情况
        if (frame.image.empty() || frame.image.type() != CV_8UC3 || !converter_ready ||
            !converter.processToBuffer(frame.image, encoder_->inputData(), encoder_->stride(),
                                       encoder_->verticalStride())) {
            continue;
        }
        // 调用 MPP 硬件编码器编码一帧
        std::vector<uint8_t> bytes;
        bool key_frame = false;
        if (!encoder_->encode(encode_index++, bytes, key_frame)) continue;    // 编码失败，跳过
        // 再次计算延迟（编码完成后），用于更精确的端到端延迟监控
        const int64_t encoded_now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (frame.capture_mono_ns > 0 && encoded_now_ns >= frame.capture_mono_ns) {
            last_frame_age_ms_.store((encoded_now_ns - frame.capture_mono_ns) / 1000000LL);
        }
        encoded_rate_.tick();                                                // 更新编码帧率统计

        // 构建编码数据包
        auto packet = std::make_shared<recording::EncodedPacket>();
        packet->data = std::make_shared<std::vector<uint8_t>>(std::move(bytes)); // 编码后的 H.264 数据
        // PTS 计算：优先使用采集时间，保证单调递增（回退到 last_pts + 默认时长）
        packet->pts_ns = frame.capture_mono_ns > last_pts_ns
            ? frame.capture_mono_ns : last_pts_ns + default_duration;
        packet->dts_ns = packet->pts_ns;                                     // DTS = PTS（无 B 帧）
        // 帧时长：当前 PTS 减去上一帧 PTS，首帧使用默认时长
        packet->duration_ns = last_pts_ns > 0 ? packet->pts_ns - last_pts_ns : default_duration;
        packet->key_frame = key_frame;                                       // 是否为 IDR 关键帧
        packet->frame_id = frame.frame_id;                                   // 保留帧 ID
        last_pts_ns = packet->pts_ns;                                        // 更新上一帧 PTS

        // 分发：录像器始终接收（用于录像存储）
        recorder_.submitPacket(packet);
        // 如果启用 RTSP 推流，将编码包推入 RTSP 队列
        if (config_.enable_rtsp) rtsp_queue_.push(std::move(packet));
    }
}

// rtspLoop(): RTSP 推流线程主循环
// - 从 RTSP 包队列取出 H.264 编码包
// - 管理 RTSP 连接状态：正常推送、断线重连、队列溢出重同步
// - 重连策略：从下一个 IDR 关键帧开始推送（确保客户端能正确解码）
// - 两次重连之间至少间隔 2 秒（避免频繁重试）
void EncodedMediaService::rtspLoop() {
    utils::applyThreadRuntime("rtsp_output", "rtsp-output");  // 设置线程调度策略和名称
    RtspPacketWriter writer;                                   // RTSP 写入器
    bool waiting_for_key = true;                               // 是否在等待 IDR 关键帧（初始状态需要）
    auto next_reconnect = std::chrono::steady_clock::now();    // 下次允许重连的时间点
    bool disconnected_event_sent = false;                       // 是否已发送断线事件（避免重复触发录像）
    uint64_t observed_drops = rtsp_queue_.dropCount();          // 已观察到的队列丢包计数

    // 主循环：运行中或队列非空时持续处理
    while (running_.load() || !rtsp_queue_.empty()) {
        // 检测队列溢出：丢包计数变化表示有新丢包发生
        const uint64_t current_drops = rtsp_queue_.dropCount();
        if (current_drops != observed_drops) {
            std::cerr << "[RtspOutput] packet queue overflow, dropped="
                      << (current_drops - observed_drops)
                      << ", resyncing at next IDR\n";
            observed_drops = current_drops;                                    // 更新观察值
            writer.close();                                                    // 关闭当前连接
            waiting_for_key = true;                                            // 等待下一个 IDR 帧重新同步
        }
        // 从 RTSP 队列取出编码包（200ms 超时）
        std::shared_ptr<recording::EncodedPacket> packet;
        if (!rtsp_queue_.pop(packet, std::chrono::milliseconds(200)) || !packet) continue;
        // 等待关键帧模式：跳过所有非 IDR 帧，直到遇到下一个关键帧
        if (waiting_for_key && !packet->key_frame) continue;
        // 遇到关键帧且到了允许重连的时间：尝试重新建立 RTSP 连接
        if (waiting_for_key && std::chrono::steady_clock::now() >= next_reconnect &&
            writer.open(config_, codec_header_)) {
            waiting_for_key = false;                                           // 连接恢复，退出等待模式
            disconnected_event_sent = false;                                   // 重置断线事件标记
        }
        // 仍在等待关键帧/重连，继续跳过
        if (waiting_for_key) continue;
        // 正常写入模式：推送编码包到 RTSP 流
        if (!writer.write(*packet)) {
            // 写入失败（网络断开、服务端不可达等）
            writer.close();                                                    // 关闭连接
            waiting_for_key = true;                                            // 进入等待重连模式
            next_reconnect = std::chrono::steady_clock::now() + std::chrono::seconds(2); // 2 秒后重试
            if (!disconnected_event_sent) {
                // 触发录像事件：RTSP 断线（可用于标记录像中的异常时段）
                recorder_.trigger(recording::Event{recording::EventType::RtspDisconnected,
                    packet->pts_ns, 0, 1.0F, -1});
                disconnected_event_sent = true;                                 // 避免重复触发
            }
        } else {
            // 写入成功：更新端到端延迟（采集到推流完成）
            const int64_t sent_now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (packet->pts_ns > 0 && sent_now_ns >= packet->pts_ns) {
                last_frame_age_ms_.store((sent_now_ns - packet->pts_ns) / 1000000LL);
            }
            sent_rate_.tick();                                                 // 更新推流帧率统计
        }
    }
}

}  // namespace pipeline
