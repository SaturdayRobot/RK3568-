#include "storage/event_recorder.h"

#include <algorithm>    // std::max, std::count, std::sort
#include <chrono>       // 系统时钟和高精度计时
#include <cstdio>       // C 标准 IO（兼容 FFmpeg C API）
#include <cstring>      // std::memcpy 用于复制编解码头数据
#include <filesystem>   // 目录创建、文件重命名、文件大小查询
#include <iomanip>      // std::put_time 用于格式化时间字符串
#include <iostream>     // 控制台输出调试信息
#include <sstream>      // 字符串流，用于拼接文件名和类型名称
#include <sys/statvfs.h> // statvfs 系统调用，获取磁盘空间信息

// FFmpeg C API —— 用于 MP4 复用（muxing）
extern "C" {
#include <libavformat/avformat.h> // AVFormatContext, avformat_alloc_output_context2 等
#include <libavutil/avutil.h>     // AVRational, av_rescale_q 等工具宏
#include <libavutil/mem.h>        // av_mallocz 内存分配
#include <libavutil/opt.h>        // av_dict_set 字典选项设置
}

#include "utils/thread_runtime.h" // 线程运行时配置（设置线程名称和调度策略）

namespace recording {
namespace {

// 获取当前系统真实时间的毫秒值（自 Unix 纪元起）
// 返回值: 毫秒级 Unix 时间戳
int64_t nowRealMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count(); // system_clock 返回墙上时间（可被 NTP 调整）
}

// 将毫秒时间戳转换为文件名友好的日期时间字符串（YYYYMMDD_HHMMSS 格式）
// 参数 real_ms: 真实时间的毫秒值
// 返回值: 如 "20260105_143025" 格式的时间字符串
std::string timestampForFile(int64_t real_ms) {
    const std::time_t value = static_cast<std::time_t>(real_ms / 1000); // 毫秒转秒
    std::tm local{};
    localtime_r(&value, &local); // 线程安全的本地时间转换
    std::ostringstream out;
    out << std::put_time(&local, "%Y%m%d_%H%M%S"); // 格式化为紧凑时间串
    return out.str();
}

// 将事件类型集合拼接为字符串，用于文件名或元数据
// 参数 types: 事件类型集合
// 参数 separator: 分隔符（如 "_" 用于文件名，"," 用于元数据）
// 返回值: 以分隔符连接的 事件类型名称字符串
std::string joinTypes(const std::set<EventType>& types, const char* separator) {
    std::ostringstream out;
    bool first = true;
    for (const auto type : types) {
        if (!first) out << separator; // 非首项时加入分隔符
        first = false;
        out << eventTypeName(type);   // 将枚举转为可读名称
    }
    return out.str();
}

}  // namespace

// 将 EventType 枚举转换为对应的可读字符串名称
// 参数 type: 事件类型枚举值
// 返回值: C 字符串，未知类型返回 "Unknown"
const char* eventTypeName(EventType type) {
    switch (type) {
    case EventType::Person:          return "Person";          // 人员检测事件
    case EventType::Cat:             return "Cat";             // 猫检测事件
    case EventType::Dog:             return "Dog";             // 狗检测事件
    case EventType::Fire:            return "Fire";            // 火焰检测事件
    case EventType::Smoke:           return "Smoke";           // 烟雾检测事件
    case EventType::Ppe:             return "Ppe";             // 安全防护装备事件
    case EventType::SiloVolume:      return "SiloVolume";      // 料仓容积事件
    case EventType::WasteSackLevel:  return "WasteSackLevel";  // 废料袋料位事件
    case EventType::Handle:          return "Handle";          // 手柄操作事件
    case EventType::TrafficLed:      return "TrafficLed";      // 交通信号灯事件
    case EventType::SensorAlarm:     return "SensorAlarm";     // 传感器告警事件
    case EventType::ManualTrigger:   return "Manual";          // 手动触发事件
    case EventType::RtspDisconnected: return "RtspDisconnected"; // RTSP 断开事件
    }
    return "Unknown"; // 未识别的枚举值，返回未知标识
}

// ============================================================================
// EventRecorder::Muxer 类实现 —— FFmpeg MP4 复用器
//
// 职责：将 H.264 裸流帧封装为 MP4 容器格式。
// 使用 FFmpeg 的 libavformat 库，输出分片 MP4（fragmented MP4），
// 即使录制异常中断，已写入的片段也可以播放。
// ============================================================================
class EventRecorder::Muxer {
public:
    // 析构函数：自动关闭并释放 FFmpeg 资源，不写 trailer（异常关闭）
    ~Muxer() { close(false); }

    // 打开 MP4 复用器并写入文件头
    // 参数 temp_path: 临时文件路径（以 .part 结尾，完成后重命名）
    // 参数 header: H.264 编解码头数据（SPS + PPS），写入 extradata 供解码器使用
    // 参数 width: 视频帧宽度（像素）
    // 参数 height: 视频帧高度（像素）
    // 返回值: true=打开成功并写入头信息，false=失败
    bool open(const std::string& temp_path, const std::vector<uint8_t>& header,
              int width, int height) {
        close(false); // 先关闭之前可能残留的上下文
        // 创建 MP4 格式的输出上下文
        if (avformat_alloc_output_context2(&context_, nullptr, "mp4", temp_path.c_str()) < 0 ||
            !context_) {
            return false;
        }
        // 创建视频流
        AVStream* stream = avformat_new_stream(context_, nullptr);
        if (!stream) return false;
        // 设置时间基：90000 是 MP4 标准时间基（便于精确的 PTS/DTS 计算）
        stream->time_base = AVRational{1, 90000};
        stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;       // 视频流类型
        stream->codecpar->codec_id = AV_CODEC_ID_H264;           // H.264 编码
        stream->codecpar->width = width;                         // 帧宽度
        stream->codecpar->height = height;                       // 帧高度
        // 分配 extradata 空间并复制 H.264 编解码头（SPS/PPS）
        // + AV_INPUT_BUFFER_PADDING_SIZE 是 FFmpeg 的安全填充，防止读取越界
        stream->codecpar->extradata = static_cast<uint8_t*>(
            av_mallocz(header.size() + AV_INPUT_BUFFER_PADDING_SIZE));
        if (!stream->codecpar->extradata) return false;
        std::memcpy(stream->codecpar->extradata, header.data(), header.size()); // 复制头数据
        stream->codecpar->extradata_size = static_cast<int>(header.size());     // 记录头大小

        // 打开输出文件（某些格式不需要文件 IO，通过 AVFMT_NOFILE 判断）
        if (!(context_->oformat->flags & AVFMT_NOFILE) &&
            avio_open(&context_->pb, temp_path.c_str(), AVIO_FLAG_WRITE) < 0) {
            return false;
        }
        // 设置分片 MP4 选项：每个关键帧处创建新的 moof 分片
        // frag_keyframe: 在关键帧处分片
        // empty_moov: 初始 moov 为空（适合流式写入）
        // default_base_moof: 使用默认的 moof 基准
        AVDictionary* options = nullptr;
        av_dict_set(&options, "movflags", "frag_keyframe+empty_moov+default_base_moof", 0);
        const int result = avformat_write_header(context_, &options); // 写入文件头（含 extradata）
        av_dict_free(&options); // 释放字典资源
        if (result < 0) return false;
        temp_path_ = temp_path;          // 记录临时路径
        header_written_ = true;          // 标记头已写入，允许后续写帧
        return true;
    }

    // 向 MP4 文件写入一帧编码数据
    // 参数 packet: 编码包（含数据、PTS、DTS、是否关键帧等）
    // 返回值: true=写入成功，false=失败
    bool write(const EncodedPacket& packet) {
        if (!context_ || !header_written_ || !packet.data || packet.data->empty()) return false;
        // 将第一个数据包的 PTS 作为时间基准，后续帧转为相对时间
        if (base_pts_ns_ < 0) base_pts_ns_ = packet.pts_ns;
        AVPacket output{};
        av_init_packet(&output);                     // 初始化 AVPacket
        output.data = packet.data->data();            // 指向帧数据（不拷贝，Muxer 内部会处理）
        output.size = static_cast<int>(packet.data->size()); // 帧数据大小
        output.stream_index = 0;                      // 视频流索引（始终为第一条流）
        // 时间基转换：纳秒基 -> 流时间基（90000）
        const AVRational ns_base{1, 1000000000};      // 纳秒时间基
        const AVRational stream_base{1, 90000};       // MP4 流时间基
        // pts = (当前包 pts - 基准 pts) 做时间基转换，确保不小于 0
        output.pts = av_rescale_q(std::max<int64_t>(0, packet.pts_ns - base_pts_ns_),
                                  ns_base, stream_base);
        // dts 同理（解码时间戳可能早于 pts）
        output.dts = av_rescale_q(std::max<int64_t>(0, packet.dts_ns - base_pts_ns_),
                                  ns_base, stream_base);
        // duration 最小为 1 个单位，避免为零导致播放器异常
        output.duration = std::max<int64_t>(1,
            av_rescale_q(packet.duration_ns, ns_base, stream_base));
        // 标记关键帧标志
        if (packet.key_frame) output.flags |= AV_PKT_FLAG_KEY;
        // 交错写入（对于只有一条视频流的 MP4 等同于直接写入）
        return av_interleaved_write_frame(context_, &output) >= 0;
    }

    // 关闭复用器，释放 FFmpeg 资源
    // 参数 write_trailer: true=写入 MP4 文件尾（正常关闭），false=跳过（异常中断）
    // 返回值: true=关闭成功，false=过程中有错误
    bool close(bool write_trailer) {
        if (!context_) return true; // 如果没有打开任何上下文，直接返回成功
        bool ok = true;
        if (header_written_ && write_trailer)
            ok = av_write_trailer(context_) >= 0; // 写入 moov 等尾部元数据
        if (!(context_->oformat->flags & AVFMT_NOFILE) && context_->pb) {
            if (avio_closep(&context_->pb) < 0) ok = false; // 关闭文件 IO
        }
        avformat_free_context(context_); // 释放整个 AVFormatContext
        context_ = nullptr;
        header_written_ = false;
        base_pts_ns_ = -1; // 重置时间基准
        return ok;
    }

private:
    AVFormatContext* context_ = nullptr; // FFmpeg 格式上下文（输出文件的抽象）
    bool header_written_ = false;        // 文件头是否已写入（标志位）
    int64_t base_pts_ns_ = -1;           // 时间基准（首个数据包的 PTS 纳秒值），-1 表示未设置
    std::string temp_path_;              // 临时文件路径（用于调试）
};

// ============================================================================
// EventRecorder 类实现
// ============================================================================

// 构造函数：保存配置并初始化编码包有界阻塞队列
// 参数 config: 录像器配置（包含所有可调参数）
EventRecorder::EventRecorder(RecorderConfig config)
    : config_(std::move(config)),
      // 队列容量至少为 32，取配置值和 32 的较大值
      // 使用 Bounded（有界）模式，队列满时新包会覆盖旧包（自动丢弃）
      packet_queue_(static_cast<size_t>(std::max(32, config_.packet_queue_size)),
                    pipeline::QueueMode::Bounded) {}

// 析构函数：停止录像器，等待工作线程结束
EventRecorder::~EventRecorder() { stop(); }

// 启动录像器：初始化编解码头信息、创建输出目录、启动工作线程
// 参数 codec_header: H.264 编解码头（SPS + PPS 二进制数据）
// 参数 width: 视频帧宽度（像素）
// 参数 height: 视频帧高度（像素）
// 返回值: true=启动成功或已启用配置, false=初始化失败
bool EventRecorder::start(const std::vector<uint8_t>& codec_header, int width, int height) {
    if (!config_.enabled) return true;  // 配置未启用，直接返回成功（避免无意义操作）
    if (running_.exchange(true)) return true; // 原子交换，已在运行则忽略本次调用
    codec_header_ = codec_header; // 保存编解码头数据
    width_ = width;              // 保存帧宽度
    height_ = height;            // 保存帧高度
    std::error_code error;
    // 创建输出目录（如果已存在则无影响）
    std::filesystem::create_directories(config_.output_dir, error);
    // 目录创建失败或编解码头为空则初始化失败
    if (error || codec_header_.empty()) {
        std::cerr << "[EventRecorder] initialization failed: " << error.message() << '\n';
        running_.store(false); // 恢复运行标志
        return false;
    }
    packet_queue_.reset();       // 重置队列状态（丢弃残留数据）
    cleanupRetention();          // 启动时先清理旧文件，确保存储空间充足
    worker_ = std::thread(&EventRecorder::workerLoop, this); // 启动工作线程
    return true;
}

// 停止录像器：设置停止标志、关闭队列、等待工作线程结束
void EventRecorder::stop() {
    if (!running_.exchange(false)) return; // 原子设置为 false，返回旧值判断是否已在运行中
    packet_queue_.close(); // 关闭阻塞队列，使 workerLoop 中的 pop 能正常退出
    if (worker_.joinable()) worker_.join(); // 等待工作线程结束
}

// 提交编码包到内部队列（由编码器线程调用）
// 参数 packet: 编码包的共享指针，为空则不处理
void EventRecorder::submitPacket(std::shared_ptr<EncodedPacket> packet) {
    if (running_.load() && packet) packet_queue_.push(std::move(packet)); // 移动语义避免拷贝
}

// 直接触发事件（跳过检测确认逻辑，立即尝试开始录制）
// 参数 event: 事件信息（类型、时间戳、置信度、摄像头 ID）
void EventRecorder::trigger(const Event& event) {
    if (!running_.load()) return; // 录像器未运行，忽略
    std::lock_guard<std::mutex> lock(event_mutex_);
    pending_events_.push_back(event); // 直接推入待处理事件队列
}

// 更新检测状态（带滑动窗口确认防抖）
// 参数 type: 事件类型
// 参数 detected: 当前帧的检测结果（true=检测到）
// 参数 mono_ns: 单调时钟纳秒值（用于 PTS 关联）
// 参数 real_ms: 真实时间毫秒值（用于文件命名）
// 参数 camera_id: 触发事件的摄像头 ID
void EventRecorder::updateDetection(EventType type, bool detected, int64_t mono_ns,
                                    int64_t real_ms, int camera_id) {
    if (!running_.load()) return;
    bool confirmed = false;
    {
        std::lock_guard<std::mutex> lock(event_mutex_);
        // 为该事件类型维护一个滑动窗口检测历史
        auto& history = detection_history_[type];
        history.push_back(detected); // 记录当前帧的检测结果
        // 保持窗口大小不超过 confirm_window（滑动窗口截断）
        while (history.size() > static_cast<size_t>(std::max(1, config_.confirm_window))) {
            history.pop_front(); // 丢弃最旧的记录
        }
        // 统计窗口内的正向检测次数
        const int positives = static_cast<int>(std::count(history.begin(), history.end(), true));
        // 确认条件：当前帧检测为 true 且窗口内正向次数 >= 最小确认阈值
        confirmed = detected && positives >= std::max(1, config_.confirm_min_positive);
        // 确认后生成事件并推入待处理队列
        if (confirmed) pending_events_.push_back(Event{type, mono_ns, real_ms, 1.0F, camera_id});
    }
}

// 设置录制完成回调函数
// 参数 callback: 回调函数，每次片段录制完成时调用
void EventRecorder::setCompletionCallback(CompletionCallback callback) {
    std::lock_guard<std::mutex> lock(event_mutex_);
    completion_callback_ = std::move(callback); // 移动语义，避免不必要的拷贝
}

// ============================================================================
// 工作线程主循环
//
// 工作流程：
//   1. 检查队列溢出（drop count 变化），如果溢出则放弃当前录制并清空缓冲区
//   2. 消费待处理事件队列，更新录像截止时间
//   3. 消费编码包队列，追加到环形缓冲区
//   4. 如果已请求开始录制且当前包是关键帧，则开始录制（从环形缓冲区回溯写入）
//   5. 如果正在录制中，持续写入帧，到达截止时间后完成录制
// ============================================================================
void EventRecorder::workerLoop() {
    // 设置工作线程的运行时属性（名称="event_recorder"，调度策略标签="event-record"）
    utils::applyThreadRuntime("event_recorder", "event-record");
    uint64_t observed_drops = packet_queue_.dropCount(); // 记录当前丢弃计数
    while (running_.load() || !packet_queue_.empty()) {   // 只要运行中或队列非空就继续循环
        // --- 步骤 1: 检测队列溢出 ---
        const uint64_t current_drops = packet_queue_.dropCount();
        if (current_drops != observed_drops) { // 丢弃计数变化，说明发生了溢出
            std::cerr << "[EventRecorder] packet queue overflow, dropped="
                      << (current_drops - observed_drops) << ", resetting at next IDR\n";
            observed_drops = current_drops;
            if (recording_) finishRecording(false); // 异常中断当前录制
            ring_.clear();                           // 清空环形缓冲区（数据已不连续）
        }
        // --- 步骤 2: 消费待处理事件 ---
        std::deque<Event> events;
        {
            std::lock_guard<std::mutex> lock(event_mutex_);
            events.swap(pending_events_); // 原子交换：批量取出所有待处理事件
        }
        for (const auto& event : events) processEvent(event); // 逐个处理事件（更新截止时间等）

        // --- 步骤 3: 从编码包队列中取出一个包 ---
        std::shared_ptr<EncodedPacket> packet;
        // 带超时的阻塞弹出：100ms 超时用于定期检查 running_ 状态
        if (!packet_queue_.pop(packet, std::chrono::milliseconds(100))) continue;
        if (!packet) continue;                       // 空包跳过
        ring_.push_back(packet);                     // 追加到环形缓冲区
        trimRing();                                   // 按缓存时长限制修剪多余的旧帧

        // --- 步骤 4: 尝试开始录制（需在关键帧处启动） ---
        bool started_with_snapshot = false;
        if (start_requested_ && !recording_)
            started_with_snapshot = beginRecording(packet); // 在下一个关键帧处启动录制

        // --- 步骤 5: 在录制中写入帧 ---
        if (recording_ && !started_with_snapshot) {
            if (!muxer_->write(*packet)) { // 尝试写入当前帧
                std::cerr << "[EventRecorder] packet write failed, closing damaged clip\n";
                finishRecording(false);    // 写入失败则中断录制
                continue;
            }
        }
        // --- 步骤 6: 检查是否到达录制截止时间 ---
        if (recording_ && packet->pts_ns >= deadline_mono_ns_) finishRecording(true);
    }
    // 退出循环后（stop 已调用），如果还在录制中则异常结束
    if (recording_) finishRecording(false);
}

// 处理单个事件：将事件类型加入当前片段类型集合，更新录制截止时间
// 参数 event: 待处理的事件
void EventRecorder::processEvent(const Event& event) {
    event_types_.insert(event.type); // 记录该事件类型（用于文件名和元数据）
    // 计算新截止时间 = 事件触发时刻 + 事件后录制时长（秒转纳秒）
    const int64_t post_ns = static_cast<int64_t>(std::max(0, config_.post_seconds)) * 1000000000LL;
    const int64_t new_deadline = event.capture_mono_ns + post_ns;
    if (!recording_ && !start_requested_) {
        // 首次事件：记录触发时刻和截止时间，设置开始请求标志
        trigger_mono_ns_ = event.capture_mono_ns; // 触发时刻（单调时钟纳秒）
        trigger_real_ms_ = event.capture_real_ms > 0 ? event.capture_real_ms : nowRealMs(); // 触发时刻（真实毫秒）
        deadline_mono_ns_ = new_deadline;         // 初始截止时间
        start_requested_ = true;                   // 请求在下个关键帧开始录制
    } else {
        // 已在录制中或有待处理请求：延长截止时间（取最大值）
        deadline_mono_ns_ = std::max(deadline_mono_ns_, new_deadline);
    }
}

// 开始录制：在环形缓冲区中定位起始帧，创建 Muxer 并写回历史帧
// 参数 current: 当前编码包（用于确定当前时间基准）
// 返回值: true=录制成功启动，false=未能启动（如找不到合适的关键帧）
bool EventRecorder::beginRecording(const std::shared_ptr<EncodedPacket>& current) {
    if (ring_.empty() || !current) return false;
    // 计算回溯目标时间 = 触发时刻 - 事件前录制时长（秒）
    const int64_t target = trigger_mono_ns_ -
        static_cast<int64_t>(std::max(0, config_.pre_seconds)) * 1000000000LL;
    // 在环形缓冲区中寻找最合适的关键帧作为起始帧：
    // 优先选择 <= 目标时间的关键帧，确保录制片段能覆盖事件前内容
    size_t start = ring_.size(); // 默认为 ring_.size() 表示未找到
    for (size_t i = 0; i < ring_.size(); ++i) {
        if (ring_[i]->key_frame && ring_[i]->pts_ns <= target) start = i;
    }
    // 如果找不到在目标之前的关键帧（目标时间太早），退而求其次使用最早的关键帧
    if (start == ring_.size()) {
        for (size_t i = 0; i < ring_.size(); ++i) {
            if (ring_[i]->key_frame) { start = i; break; }
        }
    }
    if (start == ring_.size()) return false; // 缓冲区中没有任何关键帧，无法开始录制

    // 生成输出文件名：时间戳 + 事件类型 + .mp4
    const std::string type_name = joinTypes(event_types_, "_"); // 如 "Person_Fire"
    const std::string base = timestampForFile(trigger_real_ms_) + "_" + type_name + ".mp4";
    final_path_ = (std::filesystem::path(config_.output_dir) / base).string(); // 完整目标路径
    std::string temp_path = final_path_ + ".part"; // 临时文件路径（写入完成后重命名）
    // 创建 Muxer 并打开临时文件
    muxer_ = std::make_unique<Muxer>();
    if (!muxer_->open(temp_path, codec_header_, width_, height_)) {
        std::cerr << "[EventRecorder] cannot open " << temp_path << '\n';
        muxer_.reset(); // 释放 Muxer
        return false;
    }
    // 计算片段开始真实时间（反推：触发时间 - 起始帧与触发帧的时间差）
    clip_start_real_ms_ = trigger_real_ms_ - (trigger_mono_ns_ - ring_[start]->pts_ns) / 1000000LL;
    // 将起始帧及其之后的环形缓冲区内容全部写入 MP4
    for (size_t i = start; i < ring_.size(); ++i) {
        if (!muxer_->write(*ring_[i])) {
            finishRecording(false); // 写入失败则中断
            return false;
        }
    }
    recording_ = true;       // 设置录制中状态
    start_requested_ = false; // 清除开始请求标志
    std::cout << "[EventRecorder] recording " << final_path_ << '\n';
    return true;
}

// 完成录制：关闭 Muxer、重命名文件、触发回调、清理状态
// 参数 complete: true=正常完成, false=异常中断
void EventRecorder::finishRecording(bool complete) {
    if (!muxer_) return; // 没有活动的 Muxer，直接返回
    const bool trailer_ok = muxer_->close(complete); // 关闭复用器（complete 时写入 MP4 trailer）
    muxer_.reset(); // 释放 Muxer 实例
    const std::string temp_path = final_path_ + ".part"; // 临时文件路径
    bool renamed = false;
    if (complete && trailer_ok) {
        // 正常完成：将 .part 临时文件重命名为正式 .mp4 文件
        std::error_code error;
        std::filesystem::rename(temp_path, final_path_, error);
        renamed = !error; // 重命名成功则 renamed=true
        if (error) std::cerr << "[EventRecorder] rename failed: " << error.message() << '\n';
    }

    // 构建录制元数据
    RecordingMetadata metadata;
    metadata.file_path = renamed ? final_path_ : temp_path; // 成功用正式路径，失败用临时路径
    metadata.event_types = joinTypes(event_types_, ",");    // 逗号分隔的事件类型列表
    metadata.start_real_ms = clip_start_real_ms_;           // 片段开始时间
    metadata.end_real_ms = nowRealMs();                      // 片段结束时间（当前时刻）
    metadata.trigger_real_ms = trigger_real_ms_;             // 事件触发时间
    metadata.complete = renamed;                             // 是否成功完成
    std::error_code error;
    metadata.file_size_bytes = std::filesystem::file_size(metadata.file_path, error); // 文件大小
    // 取出回调并调用（在锁外调用，避免回调中死锁）
    CompletionCallback callback;
    {
        std::lock_guard<std::mutex> lock(event_mutex_);
        callback = completion_callback_;
    }
    if (callback) callback(metadata); // 触发录制完成回调

    recording_ = false; // 清除录制中标志
    // 溢出或写入失败时保持事件状态，等待从下一个 IDR 帧重试录制
    const bool retry = !complete && running_.load() && deadline_mono_ns_ > 0;
    start_requested_ = retry;      // 如果需要重试，重新设置开始请求
    if (!retry) event_types_.clear(); // 不需要重试则清空事件类型集合
    if (!complete) ring_.clear();      // 异常中断时清空环形缓冲区（数据可能不连续）
    final_path_.clear();              // 清空文件路径
    if (renamed) cleanupRetention();  // 成功生成新文件后触发存储清理
}

// 修剪环形缓冲区：移除超出缓存时长的旧帧，控制内存占用
void EventRecorder::trimRing() {
    if (ring_.size() < 2) return; // 少于 2 帧无需修剪
    // 最长缓存时长 = max(cache_seconds, pre_seconds + 2) 秒，取最大值确保有足够回溯数据
    const int64_t max_ns = static_cast<int64_t>(std::max(config_.cache_seconds,
        config_.pre_seconds + 2)) * 1000000000LL;
    // 从队列头部开始移除超出时间窗口的帧
    while (ring_.size() > 1 && ring_.back()->pts_ns - ring_.front()->pts_ns > max_ns) {
        ring_.pop_front(); // 丢弃最旧的帧
    }
}

// 清理存储目录中的旧文件
// 按以下三个条件之一触发清理（先按时间排序，从最旧文件开始删除）：
//   1. 文件数量超过 max_files
//   2. 总存储空间超过 max_storage_mb
//   3. 磁盘空闲空间低于 min_free_space_mb
void EventRecorder::cleanupRetention() {
    namespace fs = std::filesystem;
    struct Entry { fs::path path; uint64_t size; fs::file_time_type time; }; // 文件条目（路径、大小、修改时间）
    std::vector<Entry> files;
    uint64_t total = 0;
    std::error_code error;
    // 遍历输出目录，收集所有 .mp4 文件及其元信息
    for (const auto& item : fs::directory_iterator(config_.output_dir, error)) {
        if (!item.is_regular_file() || item.path().extension() != ".mp4") continue; // 仅处理普通 .mp4 文件
        const uint64_t size = item.file_size(error); // 获取文件大小
        files.push_back({item.path(), size, item.last_write_time(error)}); // 记录文件条目
        total += size; // 累加总大小
    }
    // 按修改时间升序排列（最旧的文件排在最前面）
    std::sort(files.begin(), files.end(), [](const Entry& a, const Entry& b) { return a.time < b.time; });
    // 计算最大允许存储字节数
    const uint64_t max_bytes = static_cast<uint64_t>(std::max(0, config_.max_storage_mb)) * 1024 * 1024;
    // 获取当前磁盘可用空间（字节）
    auto freeBytes = [&]() -> uint64_t {
        struct statvfs info{};
        return statvfs(config_.output_dir.c_str(), &info) == 0
            ? static_cast<uint64_t>(info.f_bavail) * info.f_frsize // 可用块数 x 块大小
            : UINT64_MAX; // statvfs 失败时返回最大值，确保不会因这个条件触发清理
    };
    const uint64_t min_free = static_cast<uint64_t>(std::max(0, config_.min_free_space_mb)) * 1024 * 1024;
    size_t index = 0; // 从最旧的文件开始逐个删除
    while (index < files.size() &&
           ((config_.max_files > 0 && files.size() - index > static_cast<size_t>(config_.max_files)) || // 超过文件数限制
            (max_bytes > 0 && total > max_bytes) ||     // 超过总存储大小限制
            freeBytes() < min_free)) {                   // 磁盘空闲空间不足
        fs::remove(files[index].path, error); // 删除该文件
        if (!error) total -= files[index].size; // 成功删除后更新总大小
        ++index; // 继续下一个文件
    }
}

}  // namespace recording
