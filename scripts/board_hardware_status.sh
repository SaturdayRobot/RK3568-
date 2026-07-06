#!/bin/sh
# 硬件状态检查脚本：CPU/RAM/NPU/RGA/温度/频率/服务状态一览
# 使用 0.5 秒采样窗口计算瞬时 CPU 和 NPU 负载
set -u
# 注意：不使用 -e，部分检测命令失败不应中断整个报告

# 切换到根目录，避免旧脚本工作目录已被删除导致后续命令继承失效路径
cd /

# 辅助函数：从一组候选文件中读取第一行有效内容
read_first() {
    label=$1
    shift
    for path in "$@"; do
        if [ -r "$path" ]; then
            value=$(head -n 1 "$path" 2>/dev/null || true)
            [ -n "$value" ] && printf '%-20s %s\n' "$label" "$value" && return 0
        fi
    done
    printf '%-20s %s\n' "$label" "N/A"
}

# 获取 CPU 总 tick 和空闲 tick 的快照（解析 /proc/stat 第一行）
cpu_snapshot() {
    awk '/^cpu / {
        total = 0;
        for (i = 2; i <= 9; ++i) total += $i;
        print total, $5 + $6;   # 总 tick, 空闲 tick (idle + iowait)
        exit;
    }' /proc/stat
}

# 获取进程 CPU 时间（utime + stime）
process_ticks() {
    pid=$1
    if [ "$pid" -gt 0 ] 2>/dev/null && [ -r "/proc/$pid/stat" ]; then
        awk '{ print $14 + $15 }' "/proc/$pid/stat"
    else
        echo 0
    fi
}

# --- 查询应用主进程 PID ---
app_pid=$(systemctl show -p MainPID --value rk_video_ai.service 2>/dev/null || echo 0)
case "$app_pid" in ''|*[!0-9]*) app_pid=0 ;; esac
cpu_count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
clock_ticks=$(getconf CLK_TCK 2>/dev/null || echo 100)

# --- 采样前快照 ---
set -- $(cpu_snapshot)
cpu_total_before=$1
cpu_idle_before=$2
app_ticks_before=$(process_ticks "$app_pid")

# --- NPU 0.5 秒采样窗口 ---
npu_path=
for path in /sys/kernel/debug/rknpu/load /proc/rknpu/load; do
    if [ -r "$path" ]; then npu_path=$path; break; fi
done
npu_current=N/A
npu_peak=0
sample=0
while [ "$sample" -lt 5 ]; do
    if [ -n "$npu_path" ]; then
        current=$(head -n 1 "$npu_path" 2>/dev/null | sed -n 's/[^0-9]*\([0-9][0-9]*\)%.*/\1/p')
        if [ -n "$current" ]; then
            npu_current=$current
            [ "$current" -gt "$npu_peak" ] && npu_peak=$current
        fi
    fi
    sample=$((sample + 1))
    sleep 0.1
done

# --- 采样后快照 ---
set -- $(cpu_snapshot)
cpu_total_after=$1
cpu_idle_after=$2
app_ticks_after=$(process_ticks "$app_pid")

# --- 计算 CPU 使用率 ---
# 系统整体使用率：(非空闲 tick 增量 / 总 tick 增量) * 100%
system_cpu=$(awk -v t0="$cpu_total_before" -v t1="$cpu_total_after" \
                  -v i0="$cpu_idle_before" -v i1="$cpu_idle_after" \
    'BEGIN { dt=t1-t0; di=i1-i0; printf "%.1f", (dt>0 ? 100*(dt-di)/dt : 0) }')

# 应用进程单核视角使用率（进程 CPU 时间 / 单核可用时间）
app_cpu_one=$(awk -v p0="$app_ticks_before" -v p1="$app_ticks_after" \
                   -v hz="$clock_ticks" \
    'BEGIN { dp=p1-p0; printf "%.1f", (dp>0 ? 100*dp/(hz*0.5) : 0) }')

# 全板视角使用率（单核 / 核心数）
app_cpu_board=$(awk -v value="$app_cpu_one" -v cores="$cpu_count" \
    'BEGIN { printf "%.1f", (cores>0 ? value/cores : value) }')

# =================== 输出报告 ===================

printf '=== RK3568 status | %s ===\n' "$(date '+%F %T')"
printf '%s\n' '------------------------------------------------------------'

# --- 概览 ---
printf '%s\n' '[overview]'
printf '%-20s %s%% of %s cores\n' 'System CPU (0.5s)' "$system_cpu" "$cpu_count"
if [ "$app_pid" -gt 0 ] 2>/dev/null; then
    printf '%-20s %s%% one-core | %s%% board | PID %s\n' \
        'Application CPU' "$app_cpu_one" "$app_cpu_board" "$app_pid"
    app_rss=$(awk '/^VmRSS:/ { printf "%.1f MiB", $2/1024; exit }' "/proc/$app_pid/status" 2>/dev/null)
    printf '%-20s %s\n' 'Application RSS' "${app_rss:-N/A}"
else
    printf '%-20s %s\n' 'Application' 'not running'
fi
# 内存使用（MemTotal - MemAvailable）
awk '/MemTotal:/ { total=$2 } /MemAvailable:/ { avail=$2 }
     END { printf "%-20s %.0f / %.0f MiB (%.1f%%)\n", "Memory used", (total-avail)/1024,
                  total/1024, (total>0 ? 100*(total-avail)/total : 0) }' /proc/meminfo
awk '{ printf "%-20s %s %s %s\n", "Load average", $1, $2, $3 }' /proc/loadavg
# 磁盘使用
df -h / /userdata 2>/dev/null | awk 'NR==1 { printf "%-18s %8s %8s %6s\n", "Filesystem", "Size", "Avail", "Use" }
    NR>1 && !seen[$1]++ { printf "%-18s %8s %8s %6s  %s\n", $1, $2, $4, $5, $6 }'

# --- 硬件加速器 ---
printf '%s\n' '[accelerators]'
if [ "$npu_current" = N/A ]; then
    printf '%-20s %s\n' 'NPU load / peak' 'N/A'
else
    printf '%-20s %s%% / %s%% (0.5s)\n' 'NPU load / peak' "$npu_current" "$npu_peak"
fi
read_first 'NPU version' /sys/kernel/debug/rknpu/version /proc/rknpu/version
# RGA 2D 加速器负载
if [ -r /sys/kernel/debug/rkrga/load ]; then
    rga_load=$(sed -n 's/.*load[[:space:]]*=[[:space:]]*\([0-9][0-9]*%\).*/\1/p' \
        /sys/kernel/debug/rkrga/load | head -n 1)
    printf '%-20s %s\n' 'RGA load' "${rga_load:-N/A}"
else
    printf '%-20s %s\n' 'RGA load' 'N/A'
fi
read_first 'DDR load' /sys/class/devfreq/dmc/load
read_first 'GPU utilisation' /sys/devices/platform/fde60000.gpu/utilisation \
    /sys/class/devfreq/fde60000.gpu/load

# --- 温度 / 时钟频率 ---
printf '%s\n' '[temperature / clocks]'
for zone in /sys/class/thermal/thermal_zone*; do
    [ -d "$zone" ] || continue
    type=$(cat "$zone/type" 2>/dev/null || basename "$zone")
    raw=$(cat "$zone/temp" 2>/dev/null || echo 0)
    awk -v name="$type" -v value="$raw" 'BEGIN { printf "%-20s %.1f C\n", name, value/1000 }'
done
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    [ -r "$cpu/cpufreq/scaling_cur_freq" ] || continue
    khz=$(cat "$cpu/cpufreq/scaling_cur_freq")
    awk -v name="$(basename "$cpu")" -v value="$khz" 'BEGIN { printf "%-20s %.0f MHz\n", name, value/1000 }'
done

# --- 服务状态 ---
printf '%s\n' '[services]'
for unit in rk_video_ai rkaiq_3A mosquitto; do
    active=$(systemctl is-active "$unit.service" 2>/dev/null || true)
    enabled=$(systemctl is-enabled "$unit.service" 2>/dev/null || true)
    printf '%-20s active=%-9s boot=%s\n' "$unit" "${active:-unknown}" "${enabled:-unknown}"
done

# --- 进程排行（CPU 单核基准）---
printf '%s\n' '[top processes | 100% = one CPU core]'
ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head -n 9

# --- RGA 驱动信息 ---
if [ -r /sys/kernel/debug/rkrga/driver_version ]; then
    printf '%s\n' '[RGA driver]'
    head -n 1 /sys/kernel/debug/rkrga/driver_version
fi
