#pragma once

// =============================================================================
// inference_result_dispatcher.h —— 推理结果异步分发器
//
// 核心职责：
//   1. 接收多路视频管线的 InferenceStats 提交（submit）
//   2. 按流 ID 独立采样，控制落库频率（shouldPersist）
//   3. 基于队列丢弃情况动态调整采样窗口（自适应节流）
//   4. 落库成功后唤醒 MQTT 上传链路（notify_cb_）
//
// 线程模型：多生产者（管线线程 submit）-> 单消费者（worker_ 线程消费 queue_）
// 生命周期：start() 创建线程 -> 运行期持续 submit/消费 -> stop() join 线程
//
// InferenceDispatchPolicy 对应 INI 的 [inference_dispatch] 段：
//   inference_sample_ms 是基础采样窗口，dynamic_sample_* 控制自适应窗口的扩缩行为
// =============================================================================

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>

#include "pipeline/bounded_queue.hpp"  // 有界线程安全队列
#include "storage/sqlite_store.h"

namespace app {

// =============================================================================
// InferenceDispatchPolicy —— 推理结果分发策略
// =============================================================================
struct InferenceDispatchPolicy {
    int inference_sample_ms = 0;          // 基础采样周期（毫秒）；0=全量落库
    int queue_size = 256;                 // 队列容量；超出后自动丢弃最旧元素
    bool warn_on_drop = true;             // 丢弃时输出告警日志
    int drop_warn_interval_ms = 1000;     // 丢弃告警节流间隔
    bool dynamic_sample_enable = true;    // 按压力自适应采样开关
    int dynamic_sample_step_ms = 1000;    // 每次扩/缩窗口的步长
    int dynamic_sample_max_ms = 15000;    // 动态采样最大窗口（15秒）
};

// =============================================================================
// InferenceResultDispatcher —— 推理结果异步分发器
// =============================================================================
class InferenceResultDispatcher {
public:
    using NotifyCallback = std::function<void()>;  // 落库成功通知回调

    InferenceResultDispatcher();
    ~InferenceResultDispatcher();  // 析构自动调用 stop()

    // 禁止拷贝（含不可共享的线程和队列所有权），仅允许移动
    InferenceResultDispatcher(const InferenceResultDispatcher&) = delete;
    InferenceResultDispatcher& operator=(const InferenceResultDispatcher&) = delete;

    // 启动异步分发线程；store 为外部存储（非拥有指针），policy 为拷贝语义快照
    bool start(data_lifecycle::SqliteStore* store,
               const InferenceDispatchPolicy& policy,
               NotifyCallback notify_cb);

    // 停止线程并释放资源（幂等）
    void stop();

    // 提交推理统计到异步队列（多生产者线程安全）
    void submit(int stream_id, const data_lifecycle::InferenceStats& stats);

    // 运行时指标查询
    std::uint64_t droppedCount() const;   // 累计丢弃数
    int currentSampleMs() const;          // 当前生效的采样窗口

private:
    struct Item {
        int stream_id = 0;
        data_lifecycle::InferenceStats stats;
    };

    // 按流 ID 独立判定是否应落库（采样节流核心）
    bool shouldPersist(data_lifecycle::InferenceStats& stats);

    // 工作线程主循环：出队 -> 采样判定 -> 落库 -> 自适应调整
    void processLoop();

    // ---- 线程控制 ----
    std::atomic<bool> running_{false};
    std::unique_ptr<pipeline::BoundedQueue<Item>> queue_;
    std::thread worker_;

    // ---- 外部依赖 ----
    data_lifecycle::SqliteStore* store_ = nullptr;  // 非拥有指针
    InferenceDispatchPolicy policy_{};
    NotifyCallback notify_cb_;

    // ---- 采样状态 ----
    std::unordered_map<int, std::int64_t> last_persist_ts_ms_;  // key=stream_id
    int adaptive_sample_ms_ = 0;  // 动态采样窗口；0=全量，范围 [inference_sample_ms, dynamic_sample_max_ms]

    // ---- 丢弃追踪（自适应采样 & 告警节流） ----
    std::uint64_t last_drop_total_ = 0;
    std::int64_t last_drop_warn_ms_ = 0;
    std::int64_t last_drop_activity_ms_ = 0;
};

}  // namespace app
