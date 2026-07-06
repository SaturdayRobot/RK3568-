#pragma once

#include <atomic>       // 原子操作，用于无锁状态标志
#include <chrono>       // 高精度时间（用于时间戳计算）
#include <cstdint>      // 定宽整数类型
#include <deque>        // 双端队列，用于环形缓冲区和待处理事件队列
#include <functional>   // std::function，用于录制完成回调
#include <memory>       // 智能指针 std::unique_ptr / std::shared_ptr
#include <mutex>        // 互斥锁，保护事件队列等共享数据
#include <set>          // 集合，用于记录当前片段包含的事件类型
#include <string>       // 标准字符串
#include <thread>       // std::thread，工作线程
#include <unordered_map> // 哈希表，用于每个事件类型的检测历史
#include <vector>       // 动态数组，用于编解码头信息

#include "pipeline/bounded_queue.hpp"  // 有界阻塞队列，用于编码包的生产-消费模型

namespace recording {

// ============================================================================
// EventType 枚举 —— 系统支持的事件类型
// 每个值对应一种可触发录像的检测/状态事件
// ============================================================================
enum class EventType : uint8_t {
    Person,           // 人员检测
    Cat,              // 猫检测
    Dog,              // 狗检测
    Fire,             // 火焰检测
    Smoke,            // 烟雾检测
    Ppe,              // 安全防护装备检测（PPE）
    SiloVolume,       // 料仓容积状态
    WasteSackLevel,   // 废料袋料位状态
    Handle,           // 手柄操作事件
    TrafficLed,       // 交通信号灯事件
    SensorAlarm,      // 传感器告警
    ManualTrigger,    // 手动触发（人工操作触发录像）
    RtspDisconnected, // RTSP 流断开连接事件
};

// 将 EventType 枚举值转换为可读的字符串名称
// 参数 type: 事件类型枚举值
// 返回值: 对应的事件类型名称 C 字符串（如 "Person", "Fire" 等），未知类型返回 "Unknown"
const char* eventTypeName(EventType type);

// ============================================================================
// EncodedPacket 结构体 —— 封装的已编码视频包
// 表示从编码器输出的单个 H.264/H.265 编码帧数据包
// ============================================================================
struct EncodedPacket {
    std::shared_ptr<std::vector<uint8_t>> data; // 编码后的帧数据（共享指针，避免拷贝）
    int64_t pts_ns = 0;         // 显示时间戳（Presentation TimeStamp），单位纳秒
    int64_t dts_ns = 0;         // 解码时间戳（Decode TimeStamp），单位纳秒
    int64_t duration_ns = 0;    // 帧持续时间，单位纳秒
    bool key_frame = false;     // 是否为关键帧（IDR 帧），用于作为录像切割边界
    uint64_t frame_id = 0;      // 帧序号，自增唯一标识
};

// ============================================================================
// Event 结构体 —— 事件触发信息
// 当检测到某个事件类型时生成，包含时间、置信度等元信息
// ============================================================================
struct Event {
    EventType type = EventType::ManualTrigger; // 事件类型，默认为手动触发
    int64_t capture_mono_ns = 0;  // 事件触发时刻的单调时钟纳秒值（不受系统时间调整影响）
    int64_t capture_real_ms = 0;  // 事件触发时刻的真实时间毫秒值（用于文件命名）
    float confidence = 1.0F;      // 检测置信度（0.0 ~ 1.0），1.0 表示完全确定
    int camera_id = -1;           // 触发事件的摄像头 ID，-1 表示未指定
};

// ============================================================================
// RecorderConfig 结构体 —— 事件录像器的全部配置参数
// 控制录像的前后时长、缓冲大小、确认策略、存储限制等
// ============================================================================
struct RecorderConfig {
    bool enabled = false;           // 是否启用事件录像功能（关闭时可节省资源）
    int pre_seconds = 3;            // 事件前录制时长（秒），从环形缓冲区回溯写入
    int post_seconds = 5;           // 事件后录制时长（秒），到期后停止当前录制
    int cache_seconds = 5;          // 环形缓冲区最长缓存时长（秒），控制内存占用
    int packet_queue_size = 512;    // 编码包队列容量，超过则丢弃旧包防止内存溢出
    int confirm_window = 5;         // 检测确认窗口大小（帧数），用于防抖过滤误检
    int confirm_min_positive = 3;   // 确认窗口内最少正向帧数，达到此阈值才确认为真事件
    std::string output_dir = "../data/records"; // 录像文件输出目录
    int max_files = 200;            // 最大保留文件数，超出则按时间顺序删除最旧文件
    int max_storage_mb = 4096;      // 最大存储空间（MB），超出触发清理
    int min_free_space_mb = 512;    // 最小空闲磁盘空间（MB），低于此值触发清理
};

// ============================================================================
// RecordingMetadata 结构体 —— 录制片段的元数据
// 在录制完成后生成，用于回调通知和数据库记录
// ============================================================================
struct RecordingMetadata {
    std::string file_path;          // 录像文件的完整路径
    std::string event_types;        // 该片段包含的事件类型，以逗号分隔（如 "Person,Fire"）
    int64_t start_real_ms = 0;      // 片段开始时刻的真实时间毫秒值
    int64_t end_real_ms = 0;        // 片段结束时刻的真实时间毫秒值
    int64_t trigger_real_ms = 0;    // 触发事件的真实时间毫秒值
    uint64_t file_size_bytes = 0;   // 文件大小（字节）
    bool complete = false;          // 录制是否成功完成（true=完整写入并重命名，false=异常中断）
};

// ============================================================================
// EventRecorder 类 —— 事件驱动的视频录制器
//
// 核心功能：
//   1. 维护编码包的环形缓冲区（ring buffer），持续缓存最近的视频帧
//   2. 当事件触发时，从环形缓冲区中回溯 pre_seconds 秒的帧，
//      然后继续录制 post_seconds 秒，生成一段完整的 MP4 事件片段
//   3. 使用 FFmpeg libavformat 进行 MP4 复用（muxing），支持分片模式写入
//   4. 支持检测确认机制（滑动窗口防抖），避免短暂误检触发录像
//   5. 自动管理存储空间：按文件数、总大小、空闲空间三个维度清理旧文件
//
// 线程模型：
//   - 调用方线程：调用 submitPacket() 推入编码包，调用 trigger()/updateDetection() 推入事件
//   - 内部工作线程 workerLoop()：消费编码包和事件，执行录像逻辑
// ============================================================================
class EventRecorder {
public:
    // 录制完成回调类型：当一段录像完成（无论成功或失败）时调用，传递元数据
    using CompletionCallback = std::function<void(const RecordingMetadata&)>;

    // 构造函数：传入配置，初始化编码包队列（有界阻塞队列）
    explicit EventRecorder(RecorderConfig config);
    // 析构函数：自动停止录像并等待工作线程退出
    ~EventRecorder();

    // 禁止拷贝（内部持有线程和互斥锁，不可安全拷贝）
    EventRecorder(const EventRecorder&) = delete;
    EventRecorder& operator=(const EventRecorder&) = delete;

    // 启动录像器：传入 H.264 编解码头（SPS/PPS）、视频宽高
    // 头信息用于写入 MP4 文件的 extradata，没有则无法正常生成 MP4
    // 返回 true 表示启动成功（或已启用），false 表示初始化失败
    bool start(const std::vector<uint8_t>& codec_header, int width, int height);
    // 停止录像器：设置停止标志、关闭队列、等待工作线程结束
    void stop();
    // 提交一个编码包到内部队列，由工作线程消费写入环形缓冲区
    // 参数 packet: 编码包共享指针，若为空则不处理
    void submitPacket(std::shared_ptr<EncodedPacket> packet);
    // 直接触发一个事件（跳过检测确认逻辑），立即尝试开始录制
    // 参数 event: 事件信息（类型、时间戳等）
    void trigger(const Event& event);
    // 更新某个事件类型的检测状态（带滑动窗口确认防抖）
    // 参数 type: 事件类型
    // 参数 detected: 当前帧的检测结果（true=检测到，false=未检测到）
    // 参数 capture_mono_ns: 帧的单调时钟纳秒值
    // 参数 capture_real_ms: 帧的真实时间毫秒值
    // 参数 camera_id: 摄像头 ID
    void updateDetection(EventType type, bool detected, int64_t capture_mono_ns,
                         int64_t capture_real_ms, int camera_id);
    // 设置录像完成回调，每次片段结束时回调（在录制线程中调用，注意线程安全）
    void setCompletionCallback(CompletionCallback callback);

    // 获取队列丢弃的编码包数量（用于监控队列溢出情况）
    uint64_t droppedPackets() const { return packet_queue_.dropCount(); }

private:
    // Muxer 内部类：封装 FFmpeg MP4 复用器操作（打开/写入帧/关闭）
    class Muxer;
    // 工作线程主循环：从队列中消费编码包和事件，执行录像启停逻辑
    void workerLoop();
    // 处理单个事件：更新事件类型集合、延长录像截止时间
    // 参数 event: 待处理的事件
    void processEvent(const Event& event);
    // 开始录制：从环形缓冲区中定位起止帧，创建 Muxer 并写入缓冲区中的历史帧
    // 参数 current: 当前帧（用于确定回溯基准时间）
    // 返回 true 表示录制成功开始
    bool beginRecording(const std::shared_ptr<EncodedPacket>& current);
    // 完成录制：关闭 Muxer、重命名临时文件、触发回调、清理状态
    // 参数 complete: true=正常完成（写 trailer、重命名），false=异常中断（保留 .part 文件）
    void finishRecording(bool complete);
    // 修剪环形缓冲区：按 cache_seconds 配置移除超出时间窗口的旧帧，控制内存占用
    void trimRing();
    // 清理存储目录中的旧文件：按文件数、总大小、空闲空间三个条件触发删除
    void cleanupRetention();

    RecorderConfig config_;                              // 录像器配置（含所有可调参数）
    pipeline::BoundedQueue<std::shared_ptr<EncodedPacket>> packet_queue_; // 编码包有界阻塞队列
    std::atomic<bool> running_{false};                   // 运行状态标志（原子变量，线程安全）
    std::thread worker_;                                 // 工作线程句柄

    std::mutex event_mutex_;                             // 保护事件相关数据的互斥锁
    std::deque<Event> pending_events_;                   // 待处理事件队列（由外部线程推入，工作线程消费）
    std::unordered_map<EventType, std::deque<bool>> detection_history_; // 每种事件类型的检测历史（滑动窗口）
    std::unordered_map<EventType, bool> detection_latched_; // 每种事件类型的锁存状态（防止重复触发）

    std::deque<std::shared_ptr<EncodedPacket>> ring_;    // 环形缓冲区（按 PTS 排序的编码包历史）
    std::vector<uint8_t> codec_header_;                   // H.264 编解码头（SPS + PPS 数据）
    int width_ = 0;                                      // 视频帧宽度（像素）
    int height_ = 0;                                     // 视频帧高度（像素）
    bool recording_ = false;                             // 当前是否正在录制
    bool start_requested_ = false;                       // 是否已请求开始录制（在下一个关键帧处启动）
    int64_t trigger_mono_ns_ = 0;                        // 触发事件的单调时钟纳秒值（用于回溯计算）
    int64_t deadline_mono_ns_ = 0;                       // 录制截止的单调时钟纳秒值（到达后停止录制）
    int64_t trigger_real_ms_ = 0;                        // 触发事件的真实时间毫秒值（用于文件命名）
    int64_t clip_start_real_ms_ = 0;                     // 片段开始的真实时间毫秒值（用于元数据）
    std::set<EventType> event_types_;                    // 当前片段中涉及的事件类型集合
    std::unique_ptr<Muxer> muxer_;                       // FFmpeg MP4 复用器实例（独占所有权）
    std::string final_path_;                             // 录制文件的最终目标路径
    CompletionCallback completion_callback_;              // 录制完成回调函数
};

}  // namespace recording
