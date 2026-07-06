#pragma once

#include <cstdint>   // 定宽整数类型（uint64_t, int64_t）
#include <array>     // 固定大小数组，用于存储 4 个模型的统计数据
#include <string>    // 标准字符串，用于 toJson() 返回序列化结果

namespace monitoring {

// ============================================================================
// BoardStatusSnapshot 结构体 —— 板卡状态快照
//
// 封装单次采集的所有板卡状态指标，包括：
//   - 硬件状态：CPU 温度、CPU 使用率、内存使用率、磁盘使用率
//   - 网络状态：在线状态、收发字节数
//   - 摄像头状态：外部 RTSP 流和 IMX415 相机的在线状态、帧数
//   - 模型推理统计：4 个模型的检测数量、上次/平均/最大耗时
//
// 提供 toJson() 方法将快照序列化为 JSON 格式，便于存储和网络传输。
// ============================================================================
struct BoardStatusSnapshot {
    double cpu_temperature_c = -1.0;          // CPU 温度（摄氏度），-1.0 表示读取失败或未采集
    double cpu_usage_percent = 0.0;           // CPU 使用率（百分比 0.0~100.0），通过 /proc/stat 差值计算
    double memory_usage_percent = 0.0;        // 内存使用率（百分比 0.0~100.0），通过 /proc/meminfo 差值计算
    double disk_usage_percent = 0.0;          // 磁盘使用率（百分比 0.0~100.0），通过 statvfs 系统调用获取
    bool network_online = false;              // 网络是否在线（任一非 lo 网卡 operstate 为 "up" 或 "unknown"）
    uint64_t network_rx_bytes = 0;            // 网络接收字节总数（所有非 lo 网卡的 rx_bytes 之和）
    uint64_t network_tx_bytes = 0;            // 网络发送字节总数（所有非 lo 网卡的 tx_bytes 之和）
    bool external_rtsp_online = false;        // 外部 RTSP 流是否在线（帧数有增长即为在线）
    bool imx415_online = false;               // IMX415 相机是否在线（设备存在且帧数有增长）
    uint64_t external_rtsp_frames = 0;        // 外部 RTSP 流累计帧数
    uint64_t imx415_frames = 0;               // IMX415 相机累计帧数
    std::array<uint64_t, 4> model_count{};    // 4 个模型各自检测到的对象数量（索引对应：coco/fire/ppe/spare）
    std::array<int64_t, 4> model_last_us{};   // 4 个模型各自最近一次推理耗时（微秒）
    std::array<int64_t, 4> model_average_us{};// 4 个模型各自的平均推理耗时（微秒）
    std::array<int64_t, 4> model_max_us{};    // 4 个模型各自的最大推理耗时（微秒）
    int64_t timestamp_ms = 0;                 // 快照采集时间戳（毫秒），使用系统真实时间

    // 将快照序列化为 JSON 字符串
    // 包含时间戳、温度、CPU/内存/磁盘使用率、网络状态、摄像头状态、各模型统计
    // 返回值: 紧凑格式的 JSON 字符串（无额外空白字符），浮点数保留 2 位小数
    std::string toJson() const;
};

// ============================================================================
// BoardStatusCollector 类 —— 板卡状态采集器
//
// 负责定期采集嵌入式板卡（RK3568）的各项运行状态指标：
//   1. CPU 温度：读取 /sys/class/thermal/thermal_zone*/temp（取所有温度区域最大值）
//   2. CPU 使用率：两次 sample 之间通过 /proc/stat 差值计算
//   3. 内存使用率：读取 /proc/meminfo 的 MemTotal 和 MemAvailable
//   4. 磁盘使用率：通过 statvfs("/") 获取根文件系统使用情况
//   5. 网络状态：遍历 /sys/class/net/ 下所有非 lo 接口的状态和流量统计
//   6. 摄像头状态：通过帧数增量判断各摄像头是否在线
//
// 线程模型：无内部线程，由调用方根据所需频率周期性调用 sample() 方法。
// ============================================================================
class BoardStatusCollector {
public:
    // 采集一次板卡状态快照
    // 参数 external_frames:        外部 RTSP 流的当前累计帧数
    // 参数 imx415_device_online:   IMX415 设备驱动是否在线（/dev/video 节点存在）
    // 参数 imx415_frames:          IMX415 相机的当前累计帧数
    // 参数 model_count:            4 个模型的检测计数数组（索引：coco/fire/ppe/spare）
    // 参数 model_last_us:          4 个模型的最近一次推理耗时数组（微秒）
    // 参数 model_average_us:       4 个模型的平均推理耗时数组（微秒）
    // 参数 model_max_us:           4 个模型的最大推理耗时数组（微秒）
    // 返回值: 填充完整的 BoardStatusSnapshot 结构体
    BoardStatusSnapshot sample(uint64_t external_frames,
                               bool imx415_device_online,
                               uint64_t imx415_frames,
                               const std::array<uint64_t, 4>& model_count,
                               const std::array<int64_t, 4>& model_last_us,
                               const std::array<int64_t, 4>& model_average_us,
                               const std::array<int64_t, 4>& model_max_us);

private:
    uint64_t previous_cpu_total_ = 0;       // 上一次采集的 CPU 总节拍数（user+nice+system+idle+iowait+irq+softirq+steal）
    uint64_t previous_cpu_idle_ = 0;        // 上一次采集的 CPU 空闲节拍数（idle+iowait）
    uint64_t previous_external_frames_ = 0; // 上一次采集的外部 RTSP 帧数（用于判断帧数增长）
    uint64_t previous_imx415_frames_ = 0;   // 上一次采集的 IMX415 帧数（用于判断帧数增长）
};
}  // namespace monitoring
