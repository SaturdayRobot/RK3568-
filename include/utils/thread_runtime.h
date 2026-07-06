#pragma once

#include <chrono>
#include <string>

namespace utils {

// ============================================================================
// 线程运行时调优工具
//
// 设计意图：RK3568 系统包含 10+ 个关键线程（视频捕获、AI 推理、MQTT 上传、
// 流推送、温度监控等），默认 Linux CFS 调度器会导致关键线程被非实时任务抢占，
// 产生不可预测的推理延迟和推流卡顿。
//
// 本模块提供两级调优：
//   进程级 (applyProcessRuntime)：mlockall 锁内存 + 进程 CPU 亲和性绑定
//   线程级 (applyThreadRuntime)：按角色独立设置调度策略/优先级/绑核/nice
//
// 配置模型：
//   [thread_rt]          全局默认值
//   [thread_rt_inference] 推理线程覆盖配置
//   [thread_rt_streaming] 推流线程覆盖配置
//   ...
// 每个 [thread_rt_<role>] 段可覆盖 policy / priority / nice / cpus / stack_kb。
//
// 关键策略：
//   - mlockall：防止实时线程因缺页（page fault）产生数十毫秒抖动。
//   - SCHED_FIFO：对 AI 推理、视频编码等延迟敏感线程使用实时调度。
//   - 隔离核：配合内核 isolcpus 参数，将关键线程捆绑到隔离核，
//     使其不受中断和内核线程干扰。
//   - stack_kb：RK3568 16+ 线程 × 默认 8MB 栈 = 128MB 虚拟内存，
//     对非递归的简单工作线程可降至 256KB，显著节省内存。
// ============================================================================

// 从 INI [thread_rt] 段加载全局配置；返回 true 表示配置文件可读取。
bool configureThreadRuntimeFromIni(const std::string& cfg_path);

// 对"当前线程"按角色应用线程级实时参数。
// @param role        角色名，对应 INI 中的 [thread_rt_<role>] 段
// @param thread_name 线程名（通过 pthread_setname_np 设置，便于 gdb/top 调试）
void applyThreadRuntime(const std::string& role, const std::string& thread_name = "");

// 对进程主线程应用进程级调优（mlockall / 进程整体 CPU 亲和性）。
// 应在所有线程创建之前调用。
void applyProcessRuntime(const std::string& process_name = "");

// 查询 thread runtime 是否在配置中启用。
bool threadRuntimeEnabled();

// 稳定周期节拍器：用于高频循环中降低累积漂移。
//
// 与 sleep_for 的关键区别：
//   sleep_for(period)  会将"循环体时间"累加到下一次等待中，
//                       导致实际周期 = period + loop_body_time  → 漂移。
//   PeriodicTick::sleep 基于绝对时间点 sleep_until，
//                       每次锚定 next_ 为固定步进，不累积抖动。
//
// 异常保护：若线程被长时间抢占（> 4 个周期），直接重锚到当前时间，
//           避免"补偿性连跳"导致 CPU 瞬间峰值。
class PeriodicTick {
public:
    explicit PeriodicTick(std::chrono::milliseconds period = std::chrono::milliseconds(1000));

    // 变更周期并重置下一次触发时间。
    void setPeriod(std::chrono::milliseconds period);

    // 睡眠到下一节拍点，并推进下一个节拍。
    void sleep();

private:
    std::chrono::milliseconds period_{1000};         // 固定周期
    std::chrono::steady_clock::time_point next_{};   // 下一个目标时刻（绝对时间）
};

} // namespace utils
