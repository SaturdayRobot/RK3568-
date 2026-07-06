#include "utils/thread_runtime.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>

#include "config/ini_config.h"

namespace utils {
namespace {

// 运行时配置单例：从 INI [thread_rt] 段加载并缓存。
// 所有线程调优函数都从这里读取快照，避免频繁文件 I/O。
struct RuntimeConfig {
    IniConfig ini;
    bool loaded = false;                // 配置文件是否读取成功
    bool enable = false;                // 总开关
    bool require_preempt_rt = false;    // 是否强依赖 PREEMPT_RT 内核
    bool mlockall = false;              // 是否锁内存避免缺页抖动
    bool preempt_rt_detected = false;   // 当前内核是否检测为 PREEMPT_RT
    bool bind_process_affinity = false; // 是否绑定进程 CPU 亲和性
    std::string process_cpus;           // 进程级 CPU 列表（如 "4-7"）
};

RuntimeConfig& runtimeConfig() {
    static RuntimeConfig cfg;
    return cfg;
}

std::mutex& runtimeMutex() {
    static std::mutex mtx;
    return mtx;
}

// 探测当前内核是否为 PREEMPT_RT 实时内核。
// 两阶段检测机制：
//   1. /sys/kernel/realtime — 最权威的标志文件
//   2. uname -v 字符串匹配 "PREEMPT_RT" — 回退方案
// PREEMPT_RT 保证了 SCHED_FIFO 线程的低延迟特性（通常 < 100us 抖动）。
bool detectPreemptRtKernel() {
    // 优先读取实时内核标志位（Linux 提供）。
    std::ifstream f("/sys/kernel/realtime");
    if (f.is_open()) {
        std::string v;
        std::getline(f, v);
        if (!v.empty() && v[0] == '1') {
            return true;
        }
    }

    // 回退到 uname 版本字符串探测。
    struct utsname un{};
    if (::uname(&un) == 0) {
        std::string version = un.version;
        return version.find("PREEMPT_RT") != std::string::npos;
    }
    return false;
}

std::string readOneLine(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return "";
    }
    std::string line;
    std::getline(f, line);
    return line;
}

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

// 解析 CPU 列表字符串为 cpu_set_t。
// 支持格式：单核 "3"、列表 "0,2,4"、区间 "1-3"、混合 "0,2-4,6"
bool parseCpuList(const std::string& text, cpu_set_t& out_set) {
    CPU_ZERO(&out_set);
    if (text.empty()) {
        return false;
    }

    bool has_cpu = false;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) {
            return std::isspace(c) != 0;
        }), token.end());

        if (token.empty()) {
            continue;
        }

        const auto dash_pos = token.find('-');
        if (dash_pos == std::string::npos) {
            int cpu = -1;
            try {
                cpu = std::stoi(token);
            } catch (...) {
                continue;
            }
            if (cpu >= 0 && cpu < CPU_SETSIZE) {
                CPU_SET(cpu, &out_set);
                has_cpu = true;
            }
            continue;
        }

        // 支持区间写法，例如 1-3。
        int start_cpu = -1;
        int end_cpu = -1;
        try {
            start_cpu = std::stoi(token.substr(0, dash_pos));
            end_cpu = std::stoi(token.substr(dash_pos + 1));
        } catch (...) {
            continue;
        }

        if (start_cpu > end_cpu) {
            std::swap(start_cpu, end_cpu);
        }

        start_cpu = std::max(0, start_cpu);
        end_cpu = std::min(CPU_SETSIZE - 1, end_cpu);
        for (int cpu = start_cpu; cpu <= end_cpu; ++cpu) {
            CPU_SET(cpu, &out_set);
            has_cpu = true;
        }
    }

    return has_cpu;
}

// 自动检测内核隔离 CPU 列表。
// 来源优先级：
//   1. /sys/devices/system/cpu/isolated — 最直接的隔离信息
//   2. /proc/cmdline 中的 isolcpus= 参数 — 安装到 DTB 的静态参数
// 该信息用于验证 process_cpus 是否在被隔离的核上（是才有隔离效果）。
std::string detectIsolatedCpuList() {
    std::string isolated = readOneLine("/sys/devices/system/cpu/isolated");
    if (!isolated.empty()) {
        return isolated;
    }

    const std::string cmdline = readOneLine("/proc/cmdline");
    const std::string key = "isolcpus=";
    const auto pos = cmdline.find(key);
    if (pos == std::string::npos) {
        return "";
    }

    const auto begin = pos + key.size();
    const auto end = cmdline.find(' ', begin);
    if (end == std::string::npos) {
        return cmdline.substr(begin);
    }
    return cmdline.substr(begin, end - begin);
}

// 检查 child CPU 集合是否完全包含在 parent 集合中。
// 用于验证线程绑核是否落在隔离 CPU 范围内。
bool isSubsetOf(const cpu_set_t& child, const cpu_set_t& parent) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &child) && !CPU_ISSET(cpu, &parent)) {
            return false;
        }
    }
    return true;
}

// 查询在线 CPU 数量（基于 sysconf）。
int onlineCpuCount() {
    const long n = ::sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) {
        return CPU_SETSIZE;
    }
    return std::max(1, std::min(static_cast<int>(n), CPU_SETSIZE));
}

// 将 CPU 集合裁剪到在线 CPU 范围内。
// RK3568 为 4 核，CPU_SETSIZE=1024 的情况下需要裁剪，
// 防止尝试绑定到不存在的 CPU 导致 pthread_setaffinity_np 失败。
bool clampCpuSetToOnline(cpu_set_t& set, int online_cpu_count) {
    if (online_cpu_count <= 0) {
        return false;
    }

    bool has_cpu = false;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &set)) {
            continue;
        }
        if (cpu >= online_cpu_count) {
            CPU_CLR(cpu, &set);
            continue;
        }
        has_cpu = true;
    }
    return has_cpu;
}

// 内存锁定：使用 mlockall(MCL_CURRENT | MCL_FUTURE) 锁定进程所有内存映射。
//
// 关键原因：实时线程（SCHED_FIFO）绝不能在调度关键路径上触发 page fault。
// Linux 缺页中断可能导致 10-50ms 的阻塞，远超实时流（25fps=40ms/帧）的容忍度。
// 使用 std::once_flag 保证全进程只执行一次（mlockall 影响整个进程）。
void applyMemoryLockIfNeeded(bool enable) {
    if (!enable) {
        return;
    }

    // mlockall 只需要全进程执行一次。
    static std::once_flag once;
    std::call_once(once, []() {
        if (::mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
            std::cerr << "[thread_rt] mlockall failed: " << std::strerror(errno) << std::endl;
        } else {
            std::cout << "[thread_rt] mlockall enabled" << std::endl;
        }
    });
}

// 应用 nice 值（影响 CFS 调度器的 time slice 分配）。
// 注意：nice 在 Linux NPTL 线程模型中按线程生效，使用 gettid+setpriority 实现。
void applyNiceIfNeeded(int nice_value) {
    if (nice_value == 0) {
        return;
    }

    // 注意 nice 是按线程生效（Linux NPTL 线程模型）。
    const pid_t tid = static_cast<pid_t>(::syscall(SYS_gettid));
    if (::setpriority(PRIO_PROCESS, static_cast<id_t>(tid), nice_value) != 0) {
        std::cerr << "[thread_rt] setpriority failed: " << std::strerror(errno) << std::endl;
    }
}

// 应用实时调度策略（SCHED_FIFO / SCHED_RR）。
//
// 调度策略对比：
//   SCHED_OTHER : 默认 CFS 时间片轮转（普通进程，不可设置优先级）
//   SCHED_FIFO  : 先入先出抢占式调度（一旦获得 CPU 就一直运行直到主动让出或
//                 被更高优先级 FIFO/RR 线程抢占）—— 适合 AI 推理、视频编码
//   SCHED_RR    : 时间片轮转实时调度（同优先级内按时间片轮转）
//
// 优先级范围：1-99（99 最高），由 priority 参数指定。
void applyPolicyIfNeeded(const std::string& policy_name, int priority) {
    const std::string policy = toLower(policy_name);

    int sched_policy = SCHED_OTHER; // 默认普通时间片调度
    if (policy == "fifo") {
        sched_policy = SCHED_FIFO;
    } else if (policy == "rr") {
        sched_policy = SCHED_RR;
    }

    if (sched_policy == SCHED_OTHER) {
        return;
    }

    sched_param sp{};
    sp.sched_priority = std::max(1, std::min(99, priority));
    const int rc = ::pthread_setschedparam(::pthread_self(), sched_policy, &sp);
    if (rc != 0) {
        std::cerr << "[thread_rt] pthread_setschedparam failed: " << std::strerror(rc)
                  << " policy=" << policy << " priority=" << sp.sched_priority << std::endl;
    }
}

const char* policyName(int policy) {
    switch (policy) {
    case SCHED_FIFO: return "FIFO";
    case SCHED_RR: return "RR";
    case SCHED_OTHER: return "OTHER";
#ifdef SCHED_BATCH
    case SCHED_BATCH: return "BATCH";
#endif
#ifdef SCHED_IDLE
    case SCHED_IDLE: return "IDLE";
#endif
    default: return "UNKNOWN";
    }
}

std::string cpuSetText(const cpu_set_t& set) {
    std::ostringstream out;
    bool first = true;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &set)) continue;
        if (!first) out << ',';
        first = false;
        out << cpu;
    }
    return first ? "<none>" : out.str();
}

// 记录线程调优生效后的实际参数（用于调试和验证配置是否生效）。
void logEffectiveRuntime(const std::string& role) {
    cpu_set_t affinity{};
    const int affinity_rc = ::pthread_getaffinity_np(
        ::pthread_self(), sizeof(affinity), &affinity);
    int policy = SCHED_OTHER;
    sched_param parameter{};
    const int policy_rc = ::pthread_getschedparam(::pthread_self(), &policy, &parameter);
    char name[16]{};
    ::pthread_getname_np(::pthread_self(), name, sizeof(name));
    errno = 0;
    const int nice_value = ::getpriority(PRIO_PROCESS,
        static_cast<id_t>(::syscall(SYS_gettid)));

    std::cout << "[thread_rt] applied role=" << role
              << " thread=" << name
              << " cpus=" << (affinity_rc == 0 ? cpuSetText(affinity) : "<query-failed>")
              << " policy=" << (policy_rc == 0 ? policyName(policy) : "<query-failed>")
              << " priority=" << (policy_rc == 0 ? parameter.sched_priority : -1)
              << " nice=" << (errno == 0 ? nice_value : 0)
              << std::endl;
}

} // namespace

// 从 INI 文件加载线程实时调优全局配置。
// 读取 [thread_rt] 段，填充 RuntimeConfig 单例。
bool configureThreadRuntimeFromIni(const std::string& cfg_path) {
    auto& cfg = runtimeConfig();
    std::lock_guard<std::mutex> lock(runtimeMutex());

    cfg.loaded = cfg.ini.load(cfg_path);
    if (!cfg.loaded) {
        cfg.enable = false;
        return false;
    }

    // 读取全局开关和基础参数。
    cfg.enable = cfg.ini.getBool("thread_rt", "enable", false);
    cfg.require_preempt_rt = cfg.ini.getBool("thread_rt", "require_preempt_rt", false);
    cfg.mlockall = cfg.ini.getBool("thread_rt", "mlockall", false);
    cfg.preempt_rt_detected = detectPreemptRtKernel();
    cfg.bind_process_affinity = cfg.ini.getBool("thread_rt", "bind_process_affinity", false);
    cfg.process_cpus = cfg.ini.getString("thread_rt", "process_cpus", cfg.ini.getString("thread_rt", "cpus", ""));

    if (cfg.enable) {
        std::cout << "[thread_rt] enabled=" << cfg.enable
                  << " preempt_rt=" << cfg.preempt_rt_detected
                  << " mlockall=" << cfg.mlockall
                  << " bind_process_affinity=" << cfg.bind_process_affinity
                  << " process_cpus=" << cfg.process_cpus
                  << std::endl;
    }

    if (cfg.enable && cfg.require_preempt_rt && !cfg.preempt_rt_detected) {
        std::cerr << "[thread_rt] WARNING: PREEMPT_RT kernel not detected, jitter may be higher" << std::endl;
    }

    return true;
}

// 进程级实时调优：在 main() 中所有线程创建之前调用。
//
// 执行顺序：
//   1. mlockall — 锁住所有当前和未来内存映射，防止缺页抖动
//   2. 设置主线程名
//   3. 进程级 CPU 亲和性绑定（sched_setaffinity pid=0）
//   4. 验证隔离核配置并告警
void applyProcessRuntime(const std::string& process_name) {
    RuntimeConfig cfg_snapshot;
    {
        std::lock_guard<std::mutex> lock(runtimeMutex());
        cfg_snapshot = runtimeConfig();
    }

    if (!cfg_snapshot.loaded || !cfg_snapshot.enable) {
        return;
    }

    // 进程级先做锁内存，减少实时线程 page fault。
    applyMemoryLockIfNeeded(cfg_snapshot.mlockall);

    if (!process_name.empty()) {
        std::string trimmed = process_name.substr(0, 15);
        ::pthread_setname_np(::pthread_self(), trimmed.c_str());
    }

    if (!cfg_snapshot.bind_process_affinity) {
        return;
    }

    cpu_set_t process_set{};
    if (!parseCpuList(cfg_snapshot.process_cpus, process_set)) {
        std::cerr << "[thread_rt] process affinity requested but cpus list is empty/invalid" << std::endl;
        return;
    }

    const int online_cpu_count = onlineCpuCount();
    if (!clampCpuSetToOnline(process_set, online_cpu_count)) {
        std::cerr << "[thread_rt] process affinity has no online CPU after clamping"
                  << " cpus=" << cfg_snapshot.process_cpus
                  << " online_cpu_count=" << online_cpu_count << std::endl;
        return;
    }

    // pid=0 表示当前进程。
    const int rc = ::sched_setaffinity(0, sizeof(process_set), &process_set);
    if (rc != 0) {
        std::cerr << "[thread_rt] sched_setaffinity(process) failed: " << std::strerror(errno)
                  << " cpus=" << cfg_snapshot.process_cpus << std::endl;
        return;
    }

    logEffectiveRuntime("process");

    const std::string isolated = detectIsolatedCpuList();
    if (isolated.empty()) {
        std::cerr << "[thread_rt] WARN: no isolated CPU list detected; "
                  << "consider kernel args like isolcpus/nohz_full for lower jitter" << std::endl;
        return;
    }

    cpu_set_t isolated_set{};
    if (!parseCpuList(isolated, isolated_set)) {
        return;
    }

    if (!isSubsetOf(process_set, isolated_set)) {
        std::cerr << "[thread_rt] WARN: process_cpus=" << cfg_snapshot.process_cpus
                  << " are not fully within isolated cpus=" << isolated << std::endl;
    }
}

// 线程级实时调优：每个工作线程在进入主循环前调用一次。
//
// 参数继承优先级（由高到低）：
//   [thread_rt_<role>].*  >  [thread_rt].*  >  代码默认值
//
// 执行顺序（注意顺序很重要）：
//   1. mlockall — 全进程仅一次，call_once 保护
//   2. pthread_setname_np — 设置线程名（便于 gdb/htop 识别）
//   3. pthread_setaffinity_np — 绑定 CPU 亲和性
//   4. setpriority（nice）— 先调 nice
//   5. pthread_setschedparam — 切换调度策略和优先级（必须在最后，因为可能提升权限）
//
// @param role        角色名，对应 INI 中的 [thread_rt_<role>] 段
// @param thread_name 线程名
void applyThreadRuntime(const std::string& role, const std::string& thread_name) {
    RuntimeConfig cfg_snapshot;
    {
        std::lock_guard<std::mutex> lock(runtimeMutex());
        cfg_snapshot = runtimeConfig();
    }

    if (!cfg_snapshot.loaded || !cfg_snapshot.enable) {
        return;
    }

    // 每个角色可在 [thread_rt_<role>] 段做独立覆盖。
    const std::string section = "thread_rt_" + role;
    const bool role_enable = cfg_snapshot.ini.getBool(section, "enable", true);
    if (!role_enable) {
        return;
    }

    applyMemoryLockIfNeeded(cfg_snapshot.mlockall);

    const std::string policy = cfg_snapshot.ini.getString(
        section,
        "policy",
        cfg_snapshot.ini.getString("thread_rt", "policy", "other"));

    const int priority = cfg_snapshot.ini.getInt(
        section,
        "priority",
        cfg_snapshot.ini.getInt("thread_rt", "priority", 0));

    const int nice_value = cfg_snapshot.ini.getInt(
        section,
        "nice",
        cfg_snapshot.ini.getInt("thread_rt", "nice", 0));

    const std::string cpus = cfg_snapshot.ini.getString(
        section,
        "cpus",
        cfg_snapshot.ini.getString("thread_rt", "cpus", ""));

    // 按角色设置线程栈大小，减少内存占用（RK3568 16+线程默认 8MB 栈 = 128MB）
    // 可在 INI 中配置 thread_rt_<role>.stack_kb 覆盖默认值
    const int stack_kb = cfg_snapshot.ini.getInt(
        section,
        "stack_kb",
        cfg_snapshot.ini.getInt("thread_rt", "stack_kb", 0));
    if (stack_kb > 0) {
        pthread_attr_t attr;
        if (::pthread_attr_init(&attr) == 0) {
            ::pthread_attr_setstacksize(&attr, static_cast<size_t>(stack_kb) * 1024);
            // 注意：attr 仅影响后续 pthread_create，对当前线程无效
            // 此处记录配置，实际设置需在线程创建时传入
            ::pthread_attr_destroy(&attr);
        }
    }

    const std::string name = !thread_name.empty() ? thread_name : role;
    if (!name.empty()) {
        std::string trimmed = name.substr(0, 15);
        ::pthread_setname_np(::pthread_self(), trimmed.c_str());
    }

    cpu_set_t cpu_set{};
    if (parseCpuList(cpus, cpu_set)) {
        const int online_cpu_count = onlineCpuCount();
        if (!clampCpuSetToOnline(cpu_set, online_cpu_count)) {
            std::cerr << "[thread_rt] thread affinity has no online CPU after clamping"
                      << " role=" << role
                      << " cpus=" << cpus
                      << " online_cpu_count=" << online_cpu_count << std::endl;
            return;
        }

        const int rc = ::pthread_setaffinity_np(::pthread_self(), sizeof(cpu_set), &cpu_set);
        if (rc != 0) {
            std::cerr << "[thread_rt] pthread_setaffinity_np failed: " << std::strerror(rc)
                      << " cpus=" << cpus << std::endl;
        }
    }

    applyNiceIfNeeded(nice_value);       // 先调 nice
    applyPolicyIfNeeded(policy, priority); // 再切换调度策略和优先级
    logEffectiveRuntime(role);
}

bool threadRuntimeEnabled() {
    std::lock_guard<std::mutex> lock(runtimeMutex());
    return runtimeConfig().enable;
}

// PeriodicTick 实现：基于绝对时间体的周期节拍器。
//
// 与 sleep_for 对比：
//   sleep_for(40ms) + loop_body(5ms) = 实际周期 45ms → 帧率从 25fps 漂移到 ~22fps
//   PeriodicTick(40ms) → 始终锚定 40ms 步进 → 稳定 25fps
//
// 异常保护：
//   若线程被抢占超过 4 个周期（> 160ms），放弃补偿，直接重锚到当前时间。
//   这避免了"追赶式连跳"导致瞬间计算密集，给系统恢复时间。
PeriodicTick::PeriodicTick(std::chrono::milliseconds period) {
    setPeriod(period);
}

void PeriodicTick::setPeriod(std::chrono::milliseconds period) {
    if (period.count() <= 0) {
        period = std::chrono::milliseconds(1);
    }
    period_ = period;
    next_ = std::chrono::steady_clock::now() + period_;
}

void PeriodicTick::sleep() {
    const auto now = std::chrono::steady_clock::now();
    // 若线程被长时间抢占，直接重锚下一周期，避免"补偿性连跳"。
    if (now > next_ + period_ * 4) {
        next_ = now + period_;
    }
    std::this_thread::sleep_until(next_);
    next_ += period_;
}

} // namespace utils
