/**
 * @file frame_hub.cpp
 * @brief 多路视频成品帧汇聚中心
 *
 * FrameHub 不是采集器，也不是推流器；它更像一个"多路最新帧公告栏"：
 *
 * - 各条视频管线在完成本帧处理后，把最终成品图 update() 进来；
 * - MosaicStreamPipeline 再从这里按源读取最新帧并拼图；
 * - waitForNew()/seq() 提供轻量通知机制，避免 mosaic 线程死循环抢锁。
 */
#include "pipeline/frame_hub.h"

#include <cstdlib>

namespace pipeline {

// update(): 上游视频管线线程将处理完成的成品帧写入指定源通道
// - source: 帧来源枚举，标识该帧属于哪路视频源（如通道A、通道B）
// - frame: 处理完成后的 cv::Mat 图像，使用移动语义避免拷贝
// - timestamp: 系统时钟时间戳，记录帧生成时刻
// - capture_mono_ns: 单调时钟采集时刻（纳秒），用于跨源时间对齐
// - overlay: 叠加层信息（检测框、帧率统计、分辨率等元数据）
void FrameHub::update(FrameSource source, cv::Mat&& frame,
                      std::chrono::system_clock::time_point timestamp,
                      int64_t capture_mono_ns,
                      FrameOverlay overlay) {
    // 空帧直接丢弃，不更新槽位
    if (frame.empty()) {
        return;
    }

    // 将 FrameSource 枚举转换为数组下标，每个源对应一个独立的槽位
    const size_t idx = static_cast<size_t>(source);
    auto& slot = slots_[idx];

    // 用 shared_ptr 托管最新帧：
    // 读取方只拿共享引用，不需要每次 clone 整张图，适合"单写多读"的最新帧场景。
    auto holder = std::make_shared<cv::Mat>(std::move(frame));

    {
        // 对当前源槽位加锁，保证写入操作的原子性
        std::lock_guard<std::mutex> lock(slot.mtx);
        slot.frame = std::move(holder);          // 更新最新帧指针（共享所有权）
        slot.timestamp = timestamp;               // 记录系统时钟时间戳
        slot.capture_mono_ns = capture_mono_ns;   // 记录单调时钟纳秒时间戳
        slot.overlay = std::move(overlay);        // 更新叠加层信息（检测框等）
        slot.seq++;                               // 递增帧序号，用于轻量通知机制
        slot.has_frame = true;                    // 标记该槽位已有有效帧数据
        // 将当前快照推入同步队列，供 takeSynchronized() 进行跨源时间对齐
        slot.sync_queue.push_back(FrameSnapshot{
            slot.frame, timestamp, capture_mono_ns, slot.seq, slot.overlay});
        // 维持同步队列深度上限，超出则丢弃最旧帧，防止内存无限增长
        while (slot.sync_queue.size() > sync_queue_depth_.load()) {
            slot.sync_queue.pop_front();
            ++slot.sync_overflow_drops;           // 记录因队列满而丢弃的帧数
        }
    }
    // 只要有新帧写入，就广播唤醒等待者（典型使用者是 MosaicStreamPipeline）。
    slot.cv.notify_all();//告知等待线程有新帧了-->也就是其他需要获取这个帧数据的其他应用程序线程可以来这里读取数据了
    //到这里上游视频管线线程的任务就完成了，但是对于图像来说这里是相关后处理（推流）起点，处于待发状态，由其他应用程序
    //调用FrameHub::get()来获取这个帧数据并进行后续处理（推流）了
}

// get(shared_ptr): 以共享指针形式读取指定源的最新帧，避免深拷贝
// - source: 要读取的视频源枚举
// - frame_out: 输出参数，接收最新帧的 shared_ptr（共享底层数据，零拷贝）
// - ts_out: 输出参数，接收该帧的系统时间戳
// - 返回值: true 表示读取成功，false 表示该源尚无有效帧
bool FrameHub::get(FrameSource source, std::shared_ptr<cv::Mat>& frame_out,
                   std::chrono::system_clock::time_point& ts_out) const {
    const size_t idx = static_cast<size_t>(source);
    const auto& slot = slots_[idx];

    // 加锁读取，保证与写入线程的互斥
    std::lock_guard<std::mutex> lock(slot.mtx);
    // 检查槽位是否已有有效帧数据
    if (!slot.has_frame || !slot.frame || slot.frame->empty()) {
        return false;
    }
    frame_out = slot.frame;   // 共享指针赋值，引用计数+1，底层像素数据不拷贝
    ts_out = slot.timestamp;  // 返回帧对应的时间戳
    return true;
}

// get(cv::Mat): 以 cv::Mat 引用形式读取最新帧（浅拷贝 header）
// - 返回的 cv::Mat 与槽内数据共享底层像素缓冲，适合只读消费
// - 如果下游需要长期持有该帧（可能被后续写入覆盖），应自行 clone()
bool FrameHub::get(FrameSource source, cv::Mat& frame_out,
                   std::chrono::system_clock::time_point& ts_out) const {
    std::shared_ptr<cv::Mat> ptr;
    if (!get(source, ptr, ts_out)) {
        return false;
    }
    // 这里返回的是浅拷贝 header，共享底层像素；
    // 适合下游只读消费。如果下游要长期持有并可能被后续覆盖，应自行 clone。
    frame_out = *ptr;
    return true;
}

// waitForNew(): 阻塞等待指定源的新帧到来，使用条件变量避免忙等轮询
// - source: 要等待的视频源枚举
// - last_seq: 调用方已知的最新帧序号，仅当 seq > last_seq 时返回
// - timeout: 最大等待时长，超时后返回 false
// - 返回值: true 表示在超时前等到新帧，false 表示超时
bool FrameHub::waitForNew(FrameSource source, uint64_t last_seq,
                          std::chrono::milliseconds timeout) const {
    const size_t idx = static_cast<size_t>(source);
    const auto& slot = slots_[idx];

    // 使用 unique_lock 以支持条件变量的 wait_for
    std::unique_lock<std::mutex> lock(slot.mtx);
    // 等待直到 slot.seq > last_seq（有新帧写入）或超时
    return slot.cv.wait_for(lock, timeout, [&slot, last_seq]() {
        return slot.seq > last_seq;
    });
}

// snapshot(): 批量读取多个源的最新帧元数据快照（不含像素数据的深拷贝）
// - sources: 要快照的源列表
// - out: 输出参数，每个源的 FrameSnapshot（共享像素指针、时间戳、序号等）
void FrameHub::snapshot(const std::vector<FrameSource>& sources,
                        std::vector<FrameSnapshot>& out) const {
    out.assign(sources.size(), FrameSnapshot{});
    for (size_t i = 0; i < sources.size(); ++i) {
        // 逐个源加锁读取元数据，不持有跨源锁以避免死锁
        const size_t idx = static_cast<size_t>(sources[i]);
        const auto& slot = slots_[idx];

        std::lock_guard<std::mutex> lock(slot.mtx);
        out[i].seq = slot.seq;                        // 帧序号
        out[i].timestamp = slot.timestamp;             // 系统时间戳
        out[i].capture_mono_ns = slot.capture_mono_ns; // 单调时钟纳秒
        out[i].overlay = slot.overlay;                 // 叠加层元数据
        // 仅当有有效帧时才共享像素指针，否则保持 nullptr
        if (slot.has_frame && slot.frame && !slot.frame->empty()) {
            out[i].frame = slot.frame;
        }
    }
}

// takeSynchronized(): 从两个源的同步队列中取出一对时间对齐的帧
// - first/second: 两个要同步的视频源（如彩色/深度双摄）
// - threshold_ns: 时间差阈值（纳秒），两帧的 capture_mono_ns 差值必须 <= 该值
// - out: 输出参数，包含匹配成功的两个快照及统计信息
// - 返回值: true 表示找到一对时间对齐的帧，false 表示未找到
// - 算法: 比较两个队列的队首帧时间，丢弃较旧者，直到找到一对时间差在阈值内的帧
bool FrameHub::takeSynchronized(FrameSource first, FrameSource second,
                                int64_t threshold_ns, SyncResult& out) {
    auto& left = slots_[static_cast<size_t>(first)];
    auto& right = slots_[static_cast<size_t>(second)];
    // scoped_lock 同时锁定两个槽位，避免死锁（C++17 特性）
    std::scoped_lock lock(left.mtx, right.mtx);
    // 收集此前因队列满而丢弃的帧数统计
    out.dropped_first += left.sync_overflow_drops;
    out.dropped_second += right.sync_overflow_drops;
    left.sync_overflow_drops = 0;   // 清零已统计的丢弃计数
    right.sync_overflow_drops = 0;
    // 只要两边队列都非空，就持续尝试匹配
    while (!left.sync_queue.empty() && !right.sync_queue.empty()) {
        // 计算两个队列队首帧的采集时间差
        const int64_t delta = left.sync_queue.front().capture_mono_ns -
                              right.sync_queue.front().capture_mono_ns;
        // 若时间差在阈值内，则匹配成功
        if (std::llabs(delta) <= threshold_ns) {
            out.first = std::move(left.sync_queue.front());   // 取出左侧匹配帧
            out.second = std::move(right.sync_queue.front()); // 取出右侧匹配帧
            out.delta_ns = delta;                             // 记录实际时间差
            left.sync_queue.pop_front();   // 从队列中移除已匹配帧
            right.sync_queue.pop_front();
            return true;
        }
        // 左边帧更旧：丢弃左边队首，继续尝试匹配
        if (delta < 0) {
            left.sync_queue.pop_front();
            ++out.dropped_first;  // 统计因时间不匹配丢弃的帧数
        } else {
            // 右边帧更旧：丢弃右边队首
            right.sync_queue.pop_front();
            ++out.dropped_second;
        }
    }
    return false;  // 至少有一边队列为空，无法继续匹配
}

// seq(): 查询指定源的当前帧序号，用于轻量轮询判断是否有新帧
// - 返回该源槽位的当前 seq 值，每次 update() 都会递增
uint64_t FrameHub::seq(FrameSource source) const {
    const size_t idx = static_cast<size_t>(source);
    const auto& slot = slots_[idx];
    std::lock_guard<std::mutex> lock(slot.mtx);
    return slot.seq;
}

} // namespace pipeline
