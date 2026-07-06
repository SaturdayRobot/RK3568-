// =============================================================================
// inference_result_dispatcher.cpp —— 推理结果异步分发器实现
// 核心职责：
//   1. 接收多路视频管线的 InferenceStats 提交
//   2. 按流 ID 独立采样，控制落库频率
//   3. 基于队列丢弃情况动态调整采样窗口（自适应节流）
//   4. 落库成功后唤醒 MQTT 上传链路
// 线程模型：多生产者（管线线程 submit）-> 单消费者（worker_ 线程消费）
// =============================================================================

#include "app/inference_result_dispatcher.h"

#include <chrono>
#include <iostream>
#include <utility>

#include "utils/thread_runtime.h"

namespace app {

// =============================================================================
// 构造与析构
// =============================================================================
InferenceResultDispatcher::InferenceResultDispatcher() = default;

InferenceResultDispatcher::~InferenceResultDispatcher() {
    // 析构时统一调 stop：防止忘记调用 stop() 导致工作线程访问已销毁成员
    stop();
}

// =============================================================================
// start —— 启动异步分发线程
// 幂等设计：重复调用 start 不重复创建线程
// 对策略参数做下限保护：防止 INI 中配置非法值（0/负数）导致死循环或除零
// =============================================================================
bool InferenceResultDispatcher::start(data_lifecycle::SqliteStore* store,
                                      const InferenceDispatchPolicy& policy,
                                      NotifyCallback notify_cb) {
    if (running_.load(std::memory_order_relaxed)) return true;  // 幂等
    if (store == nullptr) return false;

    store_ = store;
    policy_ = policy;

    // 策略参数下限保护：避免运行时异常
    policy_.queue_size = std::max(1, policy_.queue_size);
    policy_.drop_warn_interval_ms = std::max(200, policy_.drop_warn_interval_ms);
    policy_.dynamic_sample_step_ms = std::max(100, policy_.dynamic_sample_step_ms);
    policy_.dynamic_sample_max_ms = std::max(policy_.inference_sample_ms, policy_.dynamic_sample_max_ms);

    notify_cb_ = std::move(notify_cb);
    last_persist_ts_ms_.clear();
    adaptive_sample_ms_ = std::max(0, policy_.inference_sample_ms);

    last_drop_total_ = 0;
    last_drop_warn_ms_ = 0;
    last_drop_activity_ms_ = 0;

    // 有界队列：满时自动丢弃最旧元素，避免内存无限增长
    queue_ = std::make_unique<pipeline::BoundedQueue<Item>>(
        static_cast<size_t>(policy_.queue_size), pipeline::QueueMode::Bounded);
    queue_->reset();

    // release 屏障：确保 running_ 写入对工作线程可见后再创建线程
    running_.store(true, std::memory_order_release);
    worker_ = std::thread(&InferenceResultDispatcher::processLoop, this);
    return true;
}

// =============================================================================
// stop —— 停止工作线程并清理资源
// 幂等设计：exchange 原子地交换旧值和新值，旧值为 false 则直接返回
// =============================================================================
void InferenceResultDispatcher::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;

    // 关闭队列：唤醒阻塞在 pop() 的工作线程
    if (queue_) queue_->close();
    if (worker_.joinable()) worker_.join();

    notify_cb_ = nullptr;
    store_ = nullptr;
    queue_.reset();
}

// =============================================================================
// submit —— 向异步队列提交推理统计（多生产者线程安全）
// =============================================================================
void InferenceResultDispatcher::submit(int stream_id,
                                       const data_lifecycle::InferenceStats& stats) {
    if (!running_.load(std::memory_order_acquire) || store_ == nullptr || !queue_) return;

    Item item;
    item.stream_id = stream_id;
    item.stats = stats;
    queue_->push(std::move(item));  // 有界队列满时自动丢弃最旧，递增 dropCount
}

// =============================================================================
// droppedCount / currentSampleMs —— 运行时指标查询
// =============================================================================
std::uint64_t InferenceResultDispatcher::droppedCount() const {
    if (!queue_) return 0;
    return queue_->dropCount();
}

int InferenceResultDispatcher::currentSampleMs() const {
    return adaptive_sample_ms_;
}

// =============================================================================
// shouldPersist —— 判定该条推理统计是否应落库（采样节流）
// 按流 ID 独立计算间隔：每个视频流的采样判定互不影响
// sample_ms <= 0 代表全量落库模式
// =============================================================================
bool InferenceResultDispatcher::shouldPersist(data_lifecycle::InferenceStats& stats) {
    const int sample_ms = std::max(0, adaptive_sample_ms_);
    if (sample_ms <= 0) return true;  // 全量模式

    auto& last_ts = last_persist_ts_ms_[stats.stream_id];
    if (last_ts > 0 && (stats.ts_ms - last_ts) < sample_ms) {
        return false;  // 间隔不足，节流跳过
    }

    last_ts = stats.ts_ms;
    return true;
}

// =============================================================================
// processLoop —— 工作线程主循环
// 核心流程：出队 -> 采样判定 -> 落库 -> 动态调整采样窗口
// 每 200ms 超时也会执行一次自适应调整（即使队列为空），用于无丢弃时的缩窗恢复
// =============================================================================
void InferenceResultDispatcher::processLoop() {
    // 为当前线程设置 RT 调度参数（CPU 亲和性 + SCHED_FIFO）
    utils::applyThreadRuntime("inference_dispatch", "infer-dispatch");

    auto nowMs = []() -> std::int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    };

    // =========================================================================
    // adjustSamplingByDrop —— 根据队列丢弃情况动态调整采样窗口
    // 有丢弃 -> 放大采样窗口（降低落库频率，减轻存储压力）
    // 无丢弃持续 5s -> 逐步缩回基础值（恢复时序精度）
    // =========================================================================
    auto adjustSamplingByDrop = [this, &nowMs]() {
        if (!queue_) return;

        const std::uint64_t drop_total = queue_->dropCount();
        const std::int64_t now_ms = nowMs();

        if (drop_total > last_drop_total_) {
            // 发生新丢弃：输出节流告警
            if (policy_.warn_on_drop &&
                (last_drop_warn_ms_ == 0 ||
                 (now_ms - last_drop_warn_ms_) >= policy_.drop_warn_interval_ms)) {
                std::cerr << "[InferenceDispatch] queue drop detected"
                          << " dropped_delta=" << (drop_total - last_drop_total_)
                          << " dropped_total=" << drop_total
                          << " queue_depth=" << queue_->size() << '/' << queue_->capacity()
                          << " sample_ms=" << adaptive_sample_ms_
                          << std::endl;
                last_drop_warn_ms_ = now_ms;
            }

            last_drop_total_ = drop_total;
            last_drop_activity_ms_ = now_ms;

            // 动态放大采样窗口（不超过配置上限）
            if (policy_.dynamic_sample_enable && policy_.dynamic_sample_step_ms > 0) {
                const int baseline = std::max(0, policy_.inference_sample_ms);
                const int max_sample = std::max(baseline, policy_.dynamic_sample_max_ms);
                adaptive_sample_ms_ = std::min(max_sample,
                    adaptive_sample_ms_ + policy_.dynamic_sample_step_ms);
            }
            return;
        }

        // 无丢弃：等待 5s 冷却期后逐步恢复基础采样窗口
        if (!policy_.dynamic_sample_enable || policy_.dynamic_sample_step_ms <= 0) return;
        if (adaptive_sample_ms_ <= policy_.inference_sample_ms) return;
        if (last_drop_activity_ms_ == 0 || (now_ms - last_drop_activity_ms_) < 5000) return;

        adaptive_sample_ms_ = std::max(policy_.inference_sample_ms,
                                        adaptive_sample_ms_ - policy_.dynamic_sample_step_ms);
        last_drop_activity_ms_ = now_ms;
    };

    // ===== 主循环 =====
    for (;;) {
        Item item;
        // pop 超时 200ms：即使队列为空也能周期执行自适应调节
        if (!queue_ || !queue_->pop(item, std::chrono::milliseconds(200))) {
            adjustSamplingByDrop();
            if (!running_.load(std::memory_order_acquire)) break;
            continue;
        }

        if (store_ == nullptr) continue;

        auto stats = item.stats;
        stats.stream_id = item.stream_id;

        if (!shouldPersist(stats)) {
            adjustSamplingByDrop();
            continue;
        }

        // 落库成功 -> 唤醒 MQTT 上传链路
        if (store_->insertInferenceStats(stats) && notify_cb_) {
            notify_cb_();
        }

        adjustSamplingByDrop();
    }
}

}  // namespace app
