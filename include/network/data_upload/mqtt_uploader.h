#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "config/ini_config.h"
#include "storage/sqlite_store.h"

namespace network {

// 单个视频流的描述信息，用于定期上报至云端。
struct StreamInfo {
    std::string name;    // 流名称标识（如 "imx415"、"mosaic"）
    std::string url;     // RTSP 地址或设备路径
    bool enable = false; // 是否启用该流
};

// MQTT 上传器完整配置集。
// 涵盖连接参数、TLS 双向认证、重连策略（指数退避 + 随机抖动）、
// 流信息上报周期、健康心跳上报及设备标识等。
struct MqttConfig {
    // ---- 基础连接 ----
    bool enable = false;
    std::string host = "127.0.0.1";
    int port = 8883;                              // 默认 TLS 端口
    std::string topic = "edge/robot/data";         // 常规数据发布主题
    int qos = 1;                                  // QoS 0/1/2
    int keepalive = 30;                           // MQTT keepalive 秒数
    int publish_interval_sec = 60;                 // 数据发布轮询间隔
    std::string client_id = "rk3568_edge";

    // ---- 流信息定时上报 ----
    bool stream_info_enable = true;
    int stream_info_interval_sec = 60;
    std::string stream_info_topic = "edge/robot/streams";
    std::vector<StreamInfo> streams;              // 本设备管理的所有流

    // ---- 重连策略：指数退避 + 随机抖动 ----
    // 公式: wait_ms = min(backoff_min * 2^n, backoff_max) + random(0, jitter)
    int max_retry = 10;                           // 单条记录最大重试次数
    int retry_interval_sec = 30;                  // 失败记录冷却间隔
    int reconnect_backoff_min_ms = 1000;          // 重连退避初始值
    int reconnect_backoff_max_ms = 30000;         // 重连退避上限
    int reconnect_jitter_ms = 250;                // 随机抖动幅度（避免惊群效应）

    // ---- 设备标识 ----
    std::string device_id = "device_001";
    std::string site_id = "site_001";
    std::string fw_version = "1.0.0";

    // ---- TLS 双向认证 ----
    bool tls_enable = true;
    std::string ca_file = "ca.crt";
    std::string cert_file;                        // 客户端证书（双向认证）
    std::string key_file;                         // 客户端私钥
    bool tls_insecure = false;                    // 仅调试：跳过证书验证

    // ---- 健康心跳上报 ----
    bool health_report_enable = false;
    int health_report_interval_sec = 30;
    std::string health_report_topic = "edge/robot/health";
};

// MQTT 数据上传器：负责将 SQLite 中的离线数据批量发布至 MQTT Broker。
//
// 设计要点：
//   1. 生产者-消费者模型：业务线程写入 SQLite + notifyNewData()，
//      uploadLoop 线程批量拉取并发布；
//   2. 断线重连由 uploadLoop 驱动（非 mosquitto 内置重连），
//      实现指数退避 + 随机抖动，避免多设备同时冲击 Broker；
//   3. 遗嘱消息机制：初始化时设置 offline 遗嘱，
//      连接成功时主动发布 online，确保云端可感知设备上下线；
//   4. 多主题分流：常规数据、流信息、健康心跳分别发布到不同主题；
//   5. 重试上限保护：超 max_retry 的记录标记 dead，避免无限重试。
class MqttUploader {
public:

    MqttUploader();
    ~MqttUploader();

    // 从 INI 文件加载 MQTT 配置，填充到 MqttConfig 输出参数。
    static bool loadFromIni(const std::string& path, MqttConfig& out);

    // 初始化：保存配置、创建 mosquitto 客户端、设置遗嘱消息、首次连接。
    bool initialize(const MqttConfig& config, data_lifecycle::SqliteStore* store);

    // 启动后台上传线程（uploadLoop）。幂等：重复调用无副作用。
    void start();

    // 停止线程、断开连接、释放 mosquitto 资源。
    void stop();

    /**
     * @brief 通知有新数据待上传
     *
     * 当 SqliteStore 插入新记录时调用，唤醒 uploadLoop
     * 立即处理，而非等待固定间隔超时。
     */
    void notifyNewData();

    // 上传统计快照（原子读取所有计数器，无锁）。
    struct UploadStats {
        std::uint64_t reconnect_attempts = 0;
        std::uint64_t publish_success = 0;
        std::uint64_t publish_failed = 0;
        std::uint64_t stream_info_success = 0;
        std::uint64_t stream_info_failed = 0;
    };

    UploadStats statsSnapshot() const;

    // MQTT 回调分发入口（由 mosquitto 事件线程调用）。
    void handleConnectEvent(int rc);
    void handleDisconnectEvent(int rc);

private:
    // 建立 Broker 连接，配置 TLS，启动 mosquitto 事件循环线程。
    bool connectBroker();
    // 断开连接，停止事件循环，销毁 client，清理 mosquitto 全局状态。
    void disconnectBroker();
    // 发布到默认主题（config_.topic）。
    bool publishPayload(const std::string& payload);
    // 发布到指定主题（用于流信息、心跳等独立主题）。
    bool publishPayloadToTopic(const std::string& topic, const std::string& payload);

    // 上传主循环：连接维护 -> 流信息上报 -> 心跳上报 -> 批量发布 -> 状态回写。
    void uploadLoop();
    std::int64_t nowMs() const;

private:
    MqttConfig config_;
    data_lifecycle::SqliteStore* store_ = nullptr;   // SQLite 存储，提供 fetchPending 等接口

    std::atomic<bool> running_{false};               // 工作线程运行标志
    std::thread worker_;                             // 上传工作线程

    /// 上传循环条件变量：新数据到达时唤醒，避免固定 sleep 延迟
    std::mutex upload_mtx_;
    std::condition_variable upload_cv_;
    bool has_new_data_ = false;  ///< 新数据标志，notifyNewData() 置位，uploadLoop 消费后清除

    // mosquitto 客户端（void* 避免头文件泄露）
    void* client_ = nullptr;
    std::atomic<bool> connected_{false};             // 原子连接状态（多线程读写）
    bool loop_started_ = false;                      // mosquitto 事件循环是否已启动

    // 统计计数器（原子操作，无锁读取）
    std::atomic<std::uint64_t> reconnect_attempts_{0};
    std::atomic<std::uint64_t> publish_success_{0};
    std::atomic<std::uint64_t> publish_failed_{0};
    std::atomic<std::uint64_t> stream_info_success_{0};
    std::atomic<std::uint64_t> stream_info_failed_{0};

    std::int64_t start_ts_ms_ = 0;          ///< uploadLoop 启动时间戳（用于计算 uptime）
    std::int64_t last_health_report_ms_ = 0; ///< 上次健康心跳上报时间戳
};

} // namespace network
