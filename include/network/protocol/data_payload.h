/**
 * @file data_payload.h
 * @brief 数据信封协议定义
 *
 * 该文件定义了统一的数据载荷封装协议（DataEnvelope），
 * 用于将传感器数据、AI 推理结果等数据封装为 JSON 格式，
 * 方便通过 MQTT 等协议上传到远程服务器。
 *
 * 协议设计意图：
 *   - 所有上行数据统一走 DataEnvelope 信封，云端按 type 字段分流处理；
 *   - 不依赖第三方 JSON 库（如 nlohmann/json），手写拼接以减少依赖和固件体积；
 *   - payload_json 为已序列化的 JSON 子对象，在信封中被透视写入
 *     （即不做二次 JSON 编码），避免嵌套转义。
 *
 * 主要功能：
 * 1. JSON 字符串转义（escapeJson）
 * 2. DataEnvelope 结构定义与 toJson() 序列化
 */

#pragma once

#include <cstdint>
#include <string>
#include <sstream>

namespace network {
namespace protocol {

// 对 JSON 值字符串中的特殊字符进行转义。
// 处理: ", \, \n, \r, \t
// 注意：payload_json 调用方需自行保证 JSON 合法性，escapeJson 仅处理字符串值级转义。
inline std::string escapeJson(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (char ch : input) {
        if (ch == '\"') {
            out += "\\\"";
        } else if (ch == '\\') {
            out += "\\\\";
        } else if (ch == '\n') {
            out += "\\n";
        } else if (ch == '\r') {
            out += "\\r";
        } else if (ch == '\t') {
            out += "\\t";
        } else {
            out += ch;
        }
    }
    return out;
}

// 统一数据信封：所有上行 MQTT 消息的外层封装。
//
// 结构示例：
// {
//   "version": 1,
//   "device_id": "...",
//   "site_id": "...",
//   "fw_version": "...",
//   "type": "inference_result",
//   "ts_ms": 1234567890123,
//   "payload": { ... }   // payload_json 内联写入
// }
//
// 关键约定：payload_json 是一个已序列化的 JSON 对象/数组字符串，
// toJson() 直接将其拼入外层 JSON（不转义），因此调用方必须保证它是合法 JSON。
struct DataEnvelope {
    int version = 1;               // 协议版本号

    // ---- 设备溯源信息 ----
    std::string device_id;         // 设备唯一标识
    std::string site_id;           // 站点/部署位置标识
    std::string fw_version;        // 固件版本号

    std::string type;              // 数据类型标签（如 "inference"、"stream_info"、"health_report"）
    std::int64_t ts_ms = 0;        // 数据产生时间戳（毫秒）

    // 已序列化的 JSON 负载（如推理结果、流信息等）。
    // 在 toJson() 中直接内联写入外层 JSON，不转义。
    std::string payload_json;

    // 将信封序列化为完整 JSON 字符串。
    // 不依赖第三方 JSON 库，手动拼接。
    std::string toJson() const {
        std::ostringstream oss;
        oss << "{";
        oss << "\"version\":" << version << ",";
        oss << "\"device_id\":\"" << escapeJson(device_id) << "\",";
        oss << "\"site_id\":\"" << escapeJson(site_id) << "\",";
        oss << "\"fw_version\":\"" << escapeJson(fw_version) << "\",";
        oss << "\"type\":\"" << escapeJson(type) << "\",";
        oss << "\"ts_ms\":" << ts_ms << ",";
        if (payload_json.empty()) {
            oss << "\"payload\":{}";           // 空负载 → 空 JSON 对象
        } else {
            oss << "\"payload\":" << payload_json;  // 内联写入（不转义）
        }
        oss << "}";
        return oss.str();
    }
};

} // namespace protocol
} // namespace network
