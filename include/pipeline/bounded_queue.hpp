#pragma once  // 头文件保护宏，防止重复包含

/**
 * @file bounded_queue.hpp
 * @brief 线程安全的有界队列
 *
 * 支持两种策略：
 *   1. latest  — 仅保留最新帧，新帧入队时丢弃旧帧（低延迟）
 *   2. bounded — 固定容量队列，满时丢弃最旧帧（吞吐优先）
 *
 * 配合 condition_variable 通知消费者，消除轮询空转。
 *
 * 对应优化方案 阶段A：统一线程与队列模型
 */

// 标准库头文件
#include <chrono>               // std::chrono 时间工具，用于设置超时
#include <condition_variable>   // std::condition_variable 条件变量，消费者等待/通知
#include <cstddef>              // size_t 类型
#include <cstdint>              // uint64_t 等定宽整数
#include <mutex>                // std::mutex std::unique_lock 互斥锁
#include <queue>                // std::queue 底层 FIFO 队列容器

namespace pipeline {  // 管线命名空间

/**
 * @enum QueueMode
 * @brief 队列策略枚举
 *
 * 控制队列满时的行为。
 */
enum class QueueMode : uint8_t {
    Latest  = 0,  // 仅保留最新一帧：入队时清空旧数据后放入新帧（适合视频帧，低延迟）
    Bounded = 1,  // 固定容量 FIFO：满时丢弃最旧帧（适合编码包，保序）
};

/**
 * @class BoundedQueue
 * @brief 线程安全的有界帧队列
 * @tparam T 队列元素类型
 *
 * 特性：
 * - push 永不阻塞（满时丢弃最旧帧或覆盖）
 * - pop 支持超时等待（消除轮询空转）
 * - 内置丢帧计数器，用于可观测指标
 * - 支持关闭/重置，便于管线的启停管理
 */
template <typename T>
class BoundedQueue {
public:
    /**
     * @brief 构造函数
     * @param capacity 队列容量（latest 模式下固定为 1）
     * @param mode 队列策略（默认 Latest 低延迟模式）
     */
    explicit BoundedQueue(size_t capacity = 4, QueueMode mode = QueueMode::Latest)
        : capacity_(mode == QueueMode::Latest ? 1 : (capacity > 0 ? capacity : 1))  // Latest 模式容量强制为1
        , mode_(mode)              // 保存队列策略
    {}

    /**
     * @brief 入队（非阻塞）
     * @param item 要入队的元素（move 语义）
     *
     * Latest 模式：替换队列中唯一的元素（丢弃所有旧元素）
     * Bounded 模式：满时弹出最旧元素后放入新元素
     * 入队成功后通知一个等待的消费者。
     */
    void push(T item) {
        std::lock_guard<std::mutex> lock(mtx_);  // 加锁保护队列访问（lock_guard RAII 自动释放）
        if (mode_ == QueueMode::Latest) {
            // latest 模式：清空队列中所有旧元素，只保留最新一帧
            while (!queue_.empty()) {
                queue_.pop();        // 丢弃旧帧
                drop_count_++;       // 累计丢弃计数（用于监控指标）
            }
        } else {
            // bounded 模式：容量满时丢弃最旧帧（FIFO 头部），为新帧腾出空间
            while (queue_.size() >= capacity_) {
                queue_.pop();        // 丢弃队列头部最旧帧
                drop_count_++;       // 累计丢弃计数
            }
        }
        queue_.push(std::move(item)); // 将新元素移入队列（避免拷贝开销）
        cv_.notify_one();             // 通知一个等待的消费者线程
    }

    /**
     * @brief 出队（带超时等待）
     * @param[out] item 输出元素（move 语义赋值）
     * @param timeout 最大等待时间（毫秒），默认 200ms
     * @return true 成功取到元素；false 超时或队列已关闭
     */
    bool pop(T& item, std::chrono::milliseconds timeout = std::chrono::milliseconds(200)) {
        std::unique_lock<std::mutex> lock(mtx_);  // 使用 unique_lock 支持条件变量的 wait_for
        if (!cv_.wait_for(lock, timeout, [this]() {
                return !queue_.empty() || closed_;  // 等待条件：队列非空或已关闭
            })) {
            return false; // wait_for 返回 false → 超时
        }
        if (closed_ && queue_.empty()) {
            return false; // 队列已关闭且为空，不再返回数据
        }
        item = std::move(queue_.front());  // 取出队首元素（move 避免拷贝）
        queue_.pop();                       // 移除队首元素
        return true;
    }

    /**
     * @brief 尝试非阻塞出队
     * @param[out] item 输出元素
     * @return true 成功取到元素；false 队列为空
     */
    bool tryPop(T& item) {
        std::lock_guard<std::mutex> lock(mtx_);  // 加锁
        if (queue_.empty()) {
            return false;  // 队列为空，立即返回
        }
        item = std::move(queue_.front());  // 取出队首元素
        queue_.pop();                       // 移除
        return true;
    }

    /// 关闭队列，唤醒所有等待的消费者，之后 pop 将返回 false
    void close() {
        std::lock_guard<std::mutex> lock(mtx_);  // 加锁
        closed_ = true;                            // 设置关闭标志
        cv_.notify_all();                          // 唤醒所有等待中的消费者
    }

    /// 重置队列状态（重新启动管线时使用），清空数据并重置计数器
    void reset() {
        std::lock_guard<std::mutex> lock(mtx_);  // 加锁
        closed_ = false;                           // 清除关闭标志
        drop_count_ = 0;                           // 重置丢弃计数
        std::queue<T> empty;                       // 创建空队列
        queue_.swap(empty);                        // 交换清空原队列数据
    }

    /// 获取当前队列深度（队列中待处理的元素数量）
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);  // 加锁读取
        return queue_.size();
    }

    /// 队列是否为空
    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);  // 加锁读取
        return queue_.empty();
    }

    /// 获取累计丢帧数（整个生命周期内因队列满丢弃的元素总数）
    uint64_t dropCount() const {
        std::lock_guard<std::mutex> lock(mtx_);  // 加锁读取
        return drop_count_;
    }

    /// 获取队列容量上限
    size_t capacity() const { return capacity_; }  // const 成员，无需加锁

    /// 获取队列策略模式
    QueueMode mode() const { return mode_; }        // const 成员，无需加锁

private:
    const size_t    capacity_;       // 队列最大容量（构造后不可变）
    const QueueMode mode_;           // 队列策略模式（构造后不可变）

    mutable std::mutex          mtx_;          // 互斥锁（mutable 允许 const 方法加锁）
    std::condition_variable     cv_;           // 条件变量，用于消费者等待通知
    std::queue<T>               queue_;        // 底层 STL FIFO 队列，存储元素
    bool                        closed_     = false;  // 队列关闭标志（关闭后 pop 返回 false）
    uint64_t                    drop_count_ = 0;      // 累计丢弃元素计数
};

} // namespace pipeline
