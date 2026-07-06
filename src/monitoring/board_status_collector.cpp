#include "monitoring/board_status_collector.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/statvfs.h>

namespace {
uint64_t readUintFile(const std::filesystem::path& path) {
    std::ifstream input(path);
    uint64_t value = 0;
    input >> value;
    return value;
}

double readTemperature() {
    double maximum = -1.0;
    std::error_code error;
    const std::filesystem::path root("/sys/class/thermal");
    if (!std::filesystem::exists(root, error)) return maximum;
    for (const auto& entry : std::filesystem::directory_iterator(root, error)) {
        if (entry.path().filename().string().find("thermal_zone") != 0) continue;
        const uint64_t raw = readUintFile(entry.path() / "temp");
        const double value = raw > 1000 ? raw / 1000.0 : static_cast<double>(raw);
        if (value > -40.0 && value < 200.0 && value > maximum) maximum = value;
    }
    return maximum;
}

void readNetwork(bool& online, uint64_t& rx, uint64_t& tx) {
    online = false; rx = 0; tx = 0;
    std::error_code error;
    const std::filesystem::path root("/sys/class/net");
    if (!std::filesystem::exists(root, error)) return;
    for (const auto& entry : std::filesystem::directory_iterator(root, error)) {
        const std::string name = entry.path().filename().string();
        if (name == "lo") continue;
        std::ifstream state_file(entry.path() / "operstate");
        std::string state; state_file >> state;
        online = online || state == "up" || state == "unknown";
        rx += readUintFile(entry.path() / "statistics/rx_bytes");
        tx += readUintFile(entry.path() / "statistics/tx_bytes");
    }
}
}  // namespace

namespace monitoring {

std::string BoardStatusSnapshot::toJson() const {
    std::ostringstream output;
    output << std::fixed << std::setprecision(2)
           << "{\"timestamp_ms\":" << timestamp_ms
           << ",\"cpu_temperature_c\":" << cpu_temperature_c
           << ",\"cpu_usage_percent\":" << cpu_usage_percent
           << ",\"memory_usage_percent\":" << memory_usage_percent
           << ",\"disk_usage_percent\":" << disk_usage_percent
           << ",\"network_online\":" << (network_online ? "true" : "false")
           << ",\"network_rx_bytes\":" << network_rx_bytes
           << ",\"network_tx_bytes\":" << network_tx_bytes
           << ",\"external_rtsp_online\":" << (external_rtsp_online ? "true" : "false")
           << ",\"imx415_online\":" << (imx415_online ? "true" : "false")
           << ",\"external_rtsp_frames\":" << external_rtsp_frames
           << ",\"imx415_frames\":" << imx415_frames
           << ",\"models\":[";
    static const char* names[] = {"coco", "fire", "ppe", "spare"};
    for (size_t i = 0; i < model_count.size(); ++i) {
        if (i) output << ',';
        output << "{\"name\":\"" << names[i] << "\",\"count\":" << model_count[i]
               << ",\"last_us\":" << model_last_us[i]
               << ",\"average_us\":" << model_average_us[i]
               << ",\"max_us\":" << model_max_us[i] << '}';
    }
    output << "]}";
    return output.str();
}

BoardStatusSnapshot BoardStatusCollector::sample(uint64_t external_frames,
                                                  bool imx415_device_online,
                                                  uint64_t imx415_frames,
                                                  const std::array<uint64_t, 4>& model_count,
                                                  const std::array<int64_t, 4>& model_last_us,
                                                  const std::array<int64_t, 4>& model_average_us,
                                                  const std::array<int64_t, 4>& model_max_us) {
    BoardStatusSnapshot result;
    result.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    result.cpu_temperature_c = readTemperature();

    std::ifstream stat("/proc/stat");
    std::string cpu;
    uint64_t user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0, steal = 0;
    stat >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
    const uint64_t idle_total = idle + iowait;
    const uint64_t total = user + nice + system + idle + iowait + irq + softirq + steal;
    const uint64_t total_delta = total - previous_cpu_total_;
    const uint64_t idle_delta = idle_total - previous_cpu_idle_;
    if (previous_cpu_total_ != 0 && total_delta > 0)
        result.cpu_usage_percent = 100.0 * static_cast<double>(total_delta - idle_delta) / total_delta;
    previous_cpu_total_ = total; previous_cpu_idle_ = idle_total;

    std::ifstream meminfo("/proc/meminfo");
    std::string key, unit;
    uint64_t value = 0, mem_total = 0, mem_available = 0;
    while (meminfo >> key >> value >> unit) {
        if (key == "MemTotal:") mem_total = value;
        else if (key == "MemAvailable:") mem_available = value;
    }
    if (mem_total > 0)
        result.memory_usage_percent = 100.0 * static_cast<double>(mem_total - mem_available) / mem_total;

    struct statvfs disk{};
    if (::statvfs("/", &disk) == 0 && disk.f_blocks > 0)
        result.disk_usage_percent = 100.0 * static_cast<double>(disk.f_blocks - disk.f_bavail) / disk.f_blocks;

    readNetwork(result.network_online, result.network_rx_bytes, result.network_tx_bytes);
    result.external_rtsp_frames = external_frames;
    result.imx415_frames = imx415_frames;
    result.external_rtsp_online = external_frames > previous_external_frames_;
    result.imx415_online = imx415_device_online && imx415_frames > previous_imx415_frames_;
    result.model_count = model_count;
    result.model_last_us = model_last_us;
    result.model_average_us = model_average_us;
    result.model_max_us = model_max_us;
    previous_external_frames_ = external_frames;
    previous_imx415_frames_ = imx415_frames;
    return result;
}
}  // namespace monitoring
