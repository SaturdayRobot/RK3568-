#include "network/data_upload/mqtt_uploader.h"

#include <chrono>       // 时间库：用于时间戳计算和等待
#include <iostream>     // 标准输入输出：日志打印
#include <random>       // 随机数：重连抖动随机化
#include <sstream>      // 字符串流：高效JSON拼接

#include "utils/thread_runtime.h"           // 线程实时调优

#include "network/protocol/data_payload.h"  // 数据信封协议定义

#ifdef HAVE_MOSQUITTO
#include <mosquitto.h>  // MQTT 客户端库 (Eclipse Mosquitto C 接口)
#endif

namespace {

// 将流信息列表拼装为紧凑 JSON 字符串，作为 DataEnvelope 的 payload_json。
// 这里不依赖外部 JSON 库，手动拼接时通过 escapeJson 处理字符串转义。
std::string buildStreamInfoPayload(const std::vector<network::StreamInfo>& streams) {
    std::ostringstream oss;                                // 输出字符串流，高效拼接
    oss << "{\"streams\":[";                               // 打开 JSON 对象和 streams 数组
    for (size_t i = 0; i < streams.size(); ++i) {          // 遍历所有流配置
        if (i > 0) {
            oss << ",";                                     // 多元素分隔符
        }
        oss << "{";                                         // 单个流对象的开始
        oss << "\"name\":\"" << network::protocol::escapeJson(streams[i].name) << "\",";  // 流名称，JSON转义
        oss << "\"url\":\"" << network::protocol::escapeJson(streams[i].url) << "\",";    // 流URL，JSON转义
        oss << "\"enable\":" << (streams[i].enable ? 1 : 0); // 流使能标志（布尔→整数）
        oss << "}";                                         // 单个流对象的结束
    }
    oss << "]}";                                            // 关闭数组和 JSON 对象
    return oss.str();                                       // 返回拼接完成的 JSON 字符串
}

#ifdef HAVE_MOSQUITTO
// Broker 连接建立后的回调，交由类成员统一更新状态和订阅控制主题。
// 该回调由 mosquitto 事件循环线程调用，不可在其中执行耗时操作。
void onConnect(struct mosquitto*, void* userdata, int rc) { // rc==0 表示连接成功
    if (!userdata) {                                        // 安全检查：避免空指针解引用
        return;
    }
    auto* uploader = static_cast<network::MqttUploader*>(userdata); // 还原 MqttUploader 实例指针
    uploader->handleConnectEvent(rc);                       // 委托给成员函数处理连接事件
}

// Broker 断连后的回调，交由类成员统一置位连接状态。
// MQTT 协议规定正常断开 rc==0，异常断连 rc 为错误码。
void onDisconnect(struct mosquitto*, void* userdata, int rc) {
    if (!userdata) {                                        // 安全检查
        return;
    }
    auto* uploader = static_cast<network::MqttUploader*>(userdata); // 还原实例指针
    uploader->handleDisconnectEvent(rc);                    // 委托给成员函数处理断连事件
}
#endif

} // namespace

namespace network {

// ============================================================================
// 构造函数：使用 default，成员初始化由类内初始值保证
// ============================================================================
MqttUploader::MqttUploader() = default;

// ============================================================================
// 析构函数：先停止工作线程再释放资源，保证线程安全退出
// ============================================================================
MqttUploader::~MqttUploader() {
    stop();  // 确保线程退出、断开连接、清理 mosquitto 资源
}

// ============================================================================
// 从 INI 文件读取 MQTT 相关配置，并补齐默认值。
// 除基础连接参数外，还会读取：
//   1. 流信息定时上报配置（stream_info_*）；
//   2. 控制主题订阅配置；
//   3. 设备标识信息（用于封装 DataEnvelope）；
//   4. TLS 双向认证配置；
//   5. 重连退避参数（指数退避 + 随机抖动）；
//   6. 健康心跳上报配置。
//
// @param path  INI 配置文件路径
// @param out   输出参数，填充解析后的 MqttConfig
// @return      true: 文件读取并解析成功; false: 文件不存在或格式错误
// ============================================================================
bool MqttUploader::loadFromIni(const std::string& path, MqttConfig& out) {
    IniConfig cfg;                          // 创建 INI 解析器实例
    if (!cfg.load(path)) {                  // 加载配置文件
        return false;                       // 加载失败返回 false
    }

    // --- 基础连接配置 ---
    out.enable = cfg.getBool("mqtt", "enable", false);             // MQTT 总开关
    out.host = cfg.getString("mqtt", "host", "127.0.0.1");        // Broker 地址
    out.port = cfg.getInt("mqtt", "port", 8883);                  // Broker 端口（默认 TLS 8883）
    out.topic = cfg.getString("mqtt", "topic", "edge/robot/data"); // 默认发布主题
    out.qos = cfg.getInt("mqtt", "qos", 1);                       // QoS 等级（0/1/2）
    out.keepalive = cfg.getInt("mqtt", "keepalive", 30);          // 心跳保活间隔（秒）
    out.publish_interval_sec = cfg.getInt("mqtt", "publish_interval_sec", 60); // 发布轮询间隔
    out.client_id = cfg.getString("mqtt", "client_id", "rk3568_edge");        // MQTT 客户端ID

    // --- 重试/重连策略 ---
    out.max_retry = cfg.getInt("mqtt", "max_retry", 10);                        // 单条记录最大重试次数
    out.retry_interval_sec = cfg.getInt("mqtt", "retry_interval_sec", 30);      // 失败重试间隔（秒）
    out.reconnect_backoff_min_ms = cfg.getInt("mqtt", "reconnect_backoff_min_ms", 1000);   // 重连退避最小值（ms）
    out.reconnect_backoff_max_ms = cfg.getInt("mqtt", "reconnect_backoff_max_ms", 30000);  // 重连退避最大值（ms）
    out.reconnect_jitter_ms = cfg.getInt("mqtt", "reconnect_jitter_ms", 250);              // 重连随机抖动（ms）
    if (out.reconnect_backoff_min_ms <= 0) {
        out.reconnect_backoff_min_ms = 1000;                          // 保证最小退避值合法
    }
    if (out.reconnect_backoff_max_ms < out.reconnect_backoff_min_ms) {
        out.reconnect_backoff_max_ms = out.reconnect_backoff_min_ms;  // 保证 max >= min
    }
    if (out.reconnect_jitter_ms < 0) {
        out.reconnect_jitter_ms = 0;                                  // jitter 不允许负数
    }

    // --- TLS 双向认证配置 ---
    out.tls_enable = cfg.getBool("mqtt", "tls_enable", true);         // TLS 加密开关
    out.ca_file = cfg.getString("mqtt", "ca_file", "ca.crt");         // CA 根证书
    out.cert_file = cfg.getString("mqtt", "cert_file", "");           // 客户端证书（双向认证）
    out.key_file = cfg.getString("mqtt", "key_file", "");             // 客户端私钥
    out.tls_insecure = cfg.getBool("mqtt", "tls_insecure", false);    // 是否跳过证书验证（仅调试用）

    // --- 流信息定期上报 ---
    out.stream_info_enable = cfg.getBool("mqtt", "stream_info_enable", true);          // 流信息上报开关
    out.stream_info_interval_sec = cfg.getInt("mqtt", "stream_info_interval_sec", 60);  // 上报间隔
    out.stream_info_topic = cfg.getString("mqtt", "stream_info_topic", "edge/robot/streams"); // 上报主题

    // --- 构建 StreamInfo 列表 ---
    out.streams.clear();                                                    // 清空旧数据
    {
        StreamInfo info;                                                    // 外部 RTSP 流
        info.name = "external_rtsp";                                        // 流名称标识
        info.url = cfg.getString("video_source_external", "url", "");       // RTSP 地址
        info.enable = cfg.getBool("video_source_external", "enable", false);// 是否启用
        out.streams.push_back(info);                                        // 加入列表
    }
    {
        StreamInfo info;                                                    // IMX415 摄像头流
        info.name = "imx415";                                               // 流名称标识
        info.url = cfg.getString("video_source_imx415", "device", "");      // 设备路径
        info.enable = cfg.getBool("video_source_imx415", "enable", false);  // 是否启用
        out.streams.push_back(info);                                        // 加入列表
    }
    {
        const bool enable = cfg.getBool("mosaic_stream", "enable", false);          // 拼接流总开关
        const bool enable_rtsp = cfg.getBool("mosaic_stream", "enable_rtsp", true); // 拼接流 RTSP 开关
        const std::string url = cfg.getString("stream_output", "url", "");           // RTSP 输出地址
        if (!url.empty()) {                                                          // 仅当 URL 有效时添加
            StreamInfo info;
            info.name = "mosaic";                                                    // 流名称标识
            info.url = url;                                                          // RTSP 地址
            info.enable = enable && enable_rtsp;                                     // 双重开关控制
            out.streams.push_back(info);                                             // 加入列表
        }
    }

    // --- 设备标识信息 ---
    out.device_id = cfg.getString("device", "device_id", "device_001"); // 设备唯一ID
    out.site_id = cfg.getString("device", "site_id", "site_001");       // 站点ID
    out.fw_version = cfg.getString("device", "fw_version", "1.0.0");    // 固件版本

    // --- 健康心跳配置 ---
    out.health_report_enable = cfg.getBool("mqtt", "health_report_enable", false);          // 心跳上报开关
    out.health_report_interval_sec = cfg.getInt("mqtt", "health_report_interval_sec", 30);   // 上报间隔
    out.health_report_topic = cfg.getString("mqtt", "health_report_topic", "edge/robot/health"); // 心跳主题

    return true;  // 解析成功
}

// ============================================================================
// 初始化上传器：
//   1. 保存配置和存储指针；
//   2. 若功能关闭直接返回成功（允许系统按统一流程初始化）；
//   3. 创建 mosquitto 客户端并注册回调；
//   4. 设置遗嘱消息（Will Message），确保异常离线时云端可感知；
//   5. 尝试首次连接，失败时不阻断启动，由后台线程重试。
//
// @param config   MQTT 配置参数
// @param store    SQLite 存储指针，用于拉取待上传数据
// @return         true: 初始化成功或功能已禁用; false: 库不可用或创建失败
// ============================================================================
bool MqttUploader::initialize(const MqttConfig& config, data_lifecycle::SqliteStore* store) {
    config_ = config;                // 保存配置副本
    store_ = store;                  // 保存存储指针

    if (!config_.enable) {           // 功能禁用
        return true;                 // 视为成功，不影响系统启动
    }

#ifndef HAVE_MOSQUITTO
    std::cerr << "MqttUploader: libmosquitto not available, disabled" << std::endl;
    return false;                    // 编译时未链接 mosquitto，无法使用
#else
    mosquitto_lib_init();            // 初始化 mosquitto 全局库状态

    client_ = mosquitto_new(config_.client_id.c_str(), true, this); // 创建客户端对象（clean_session=true）
    if (!client_) {                  // 创建失败
        std::cerr << "MqttUploader: create client failed" << std::endl;
        return false;
    }

    auto* client = static_cast<mosquitto*>(client_);              // void* → mosquitto*
    mosquitto_connect_callback_set(client, onConnect);            // 注册连接成功回调
    mosquitto_disconnect_callback_set(client, onDisconnect);      // 注册断连回调
    mosquitto_reconnect_delay_set(client, 1, 30, true);           // 启用自动重连（1s-30s指数退避）
    const std::string will_topic = config_.health_report_topic.empty()
        ? config_.topic + "/status" : config_.health_report_topic; // 遗嘱主题
    const std::string will_payload = "{\"device_id\":\"" + config_.device_id +
                                     "\",\"online\":false}";       // 遗嘱消息内容：设备离线
    mosquitto_will_set(client, will_topic.c_str(),
                       static_cast<int>(will_payload.size()), will_payload.c_str(), 1, true); // 设置遗嘱消息

    if (!connectBroker()) {                                       // 首次连接尝试
        std::cerr << "MqttUploader: initial connect failed, uploader will retry in background" << std::endl;
    }
    return true;  // 初始化成功，即使首次连接失败也不阻塞
#endif
}

// ============================================================================
// 启动后台上传线程。重复调用会被 running_ 防护。
// 工作线程运行 uploadLoop()，按周期批量拉取数据库记录并通过 MQTT 发布。
// ============================================================================
void MqttUploader::start() {
    if (!config_.enable || running_) {   // 功能关闭或已在运行
        return;                          // 直接返回，避免重复启动
    }

    running_ = true;                                          // 标记运行状态
    worker_ = std::thread([this]() { uploadLoop(); });        // 创建后台工作线程
}

// ============================================================================
// 停止后台上传线程并释放 MQTT 连接资源。
// 执行流程：
//   1. 设置 running_ = false，通知线程退出
//   2. 唤醒 upload_cv_ 条件变量，避免线程在 wait_for 中阻塞
//   3. join 等待线程退出
//   4. 断开连接、销毁客户端、清理 mosquitto 库
// ============================================================================
void MqttUploader::stop() {
    if (!running_) {                    // 未在运行
        disconnectBroker();             // 仍然尝试断开（清理残留资源）
        return;
    }

    running_ = false;                   // 通知工作线程退出
    upload_cv_.notify_all();            // 唤醒 uploadLoop 使其尽快退出
    if (worker_.joinable()) {           // 线程可 join
        worker_.join();                 // 等待线程结束
    }
    disconnectBroker();                 // 断开连接并释放 mosquitto 资源
}

// ============================================================================
// 建立与 Broker 的连接，并在首次连接成功后启动 mosquitto 网络循环线程。
//
// 关键点：
//   - 已连接时直接返回 true，避免重复连接
//   - loop_started_ 确保 mosquitto_loop_start 只调用一次
//   - TLS 参数在 connect 前设置
//   - 返回 true 仅表示"当前状态可用或已进入可重连状态"，实际在线状态由 connected_ 维护
//
// @return  true: 已连接或网络循环已启动; false: 连接/启动失败
// ============================================================================
bool MqttUploader::connectBroker() {
#ifndef HAVE_MOSQUITTO
    return false;                               // 库不可用
#else
    if (!client_) {                             // 客户端对象不存在
        return false;
    }
    if (connected_.load()) {                    // 已经连接
        return true;
    }

    auto* client = static_cast<mosquitto*>(client_); // 类型还原

    // --- 配置 TLS 参数 ---
    if (config_.tls_enable) {                   // 启用了 TLS
        const int tls_rc = mosquitto_tls_set(   // 设置 TLS 证书和密钥
            client,
            config_.ca_file.empty() ? nullptr : config_.ca_file.c_str(),     // CA 证书
            nullptr,                                                          // capath（不使用）
            config_.cert_file.empty() ? nullptr : config_.cert_file.c_str(), // 客户端证书
            config_.key_file.empty() ? nullptr : config_.key_file.c_str(),    // 客户端私钥
            nullptr);                                                         // 私钥密码回调
        if (tls_rc != MOSQ_ERR_SUCCESS) {
            std::cerr << "MqttUploader: tls set failed" << std::endl;
        }
        mosquitto_tls_insecure_set(client, config_.tls_insecure); // 设置是否跳过证书验证
    }

    if (loop_started_) {                        // 网络循环已启动
        return true;
    }

    int rc = mosquitto_connect(client, config_.host.c_str(), config_.port, config_.keepalive); // 发起 TCP/TLS 连接
    if (rc != MOSQ_ERR_SUCCESS) {               // 连接失败
        return false;
    }

    rc = mosquitto_loop_start(client);          // 启动 mosquitto 内部事件循环（独立线程）
    if (rc != MOSQ_ERR_SUCCESS) {               // 事件循环启动失败
        mosquitto_disconnect(client);           // 断开连接
        return false;
    }

    loop_started_ = true;                       // 标记网络循环已启动
    return true;
#endif
}

// ============================================================================
// 断开连接并销毁 mosquitto 客户端对象。
// 注意：这里负责调用 mosquitto_lib_cleanup，对应 initialize 中的 lib_init，
// 调用次数需匹配，否则会导致资源泄漏。
// ============================================================================
void MqttUploader::disconnectBroker() {
#ifdef HAVE_MOSQUITTO
    connected_.store(false);                    // 原子操作：标记离线

    if (client_) {                              // 有有效客户端对象
        auto* client = static_cast<mosquitto*>(client_);
        if (loop_started_) {                    // 网络循环正在运行
            mosquitto_loop_stop(client, true);  // 停止事件循环（阻塞等待完成）
            loop_started_ = false;              // 重置标志
        }
        mosquitto_disconnect(client);           // 断开 MQTT 连接
        mosquitto_destroy(client);              // 销毁客户端对象，释放内存
        client_ = nullptr;                      // 置空指针
    }

    mosquitto_lib_cleanup();                    // 清理 mosquitto 全局状态
#endif
}

// ============================================================================
// 发布业务数据到默认数据主题（config_.topic）。
// 直接调用 mosquitto_publish，不在此处做 DataEnvelope 封装（由调用方负责）。
//
// @param payload  JSON 字符串负载
// @return         true: 发布成功; false: 发布失败或未连接
// ============================================================================
bool MqttUploader::publishPayload(const std::string& payload) {
#ifndef HAVE_MOSQUITTO
    return false;
#else
    if (!client_ || !connected_.load()) {       // 客户端无效或未连接
        return false;                           // 直接返回失败
    }

    const int rc = mosquitto_publish(           // 发布消息到 Broker
        static_cast<mosquitto*>(client_),       // 客户端句柄
        nullptr,                                // msg_id 输出（不需要）
        config_.topic.c_str(),                  // 目标主题
        static_cast<int>(payload.size()),       // 消息长度
        payload.c_str(),                        // 消息内容
        config_.qos,                            // QoS 等级
        false);                                 // retain=false
    if (rc == MOSQ_ERR_NO_CONN || rc == MOSQ_ERR_CONN_LOST) { // 连接已丢失
        connected_.store(false);                // 原子更新连接状态
    }
    if (rc == MOSQ_ERR_SUCCESS) {               // 发布成功
        publish_success_.fetch_add(1, std::memory_order_relaxed); // 原子递增成功计数
    } else {                                    // 发布失败
        publish_failed_.fetch_add(1, std::memory_order_relaxed);  // 原子递增失败计数
    }
    return rc == MOSQ_ERR_SUCCESS;              // 返回发布结果
#endif
}

// ============================================================================
// 发布数据到指定主题（用于 stream_info 与常规数据分主题上报）。
// 与 publishPayload 逻辑一致，但使用自定义 topic，并统计 stream_info 计数器。
//
// @param topic   目标主题路径
// @param payload JSON 字符串负载
// @return        true: 发布成功; false: 失败
// ============================================================================
bool MqttUploader::publishPayloadToTopic(const std::string& topic, const std::string& payload) {
#ifndef HAVE_MOSQUITTO
    return false;
#else
    if (!client_ || !connected_.load()) {       // 客户端无效或未连接
        return false;
    }

    const int rc = mosquitto_publish(           // 发布消息
        static_cast<mosquitto*>(client_),       // 客户端句柄
        nullptr,                                // msg_id
        topic.c_str(),                          // 自定义主题
        static_cast<int>(payload.size()),       // 消息长度
        payload.c_str(),                        // 消息内容
        config_.qos,                            // QoS
        false);                                 // retain=false
    if (rc == MOSQ_ERR_NO_CONN || rc == MOSQ_ERR_CONN_LOST) { // 连接丢失
        connected_.store(false);                // 标记离线
    }
    if (rc == MOSQ_ERR_SUCCESS) {               // 成功
        stream_info_success_.fetch_add(1, std::memory_order_relaxed); // 递增流信息成功计数
    } else {                                    // 失败
        stream_info_failed_.fetch_add(1, std::memory_order_relaxed);  // 递增流信息失败计数
    }
    return rc == MOSQ_ERR_SUCCESS;              // 返回结果
#endif
}

// ============================================================================
// 后台上传主循环 —— MQTT 上传器的核心调度逻辑：
//
// 启动后持续运行，直到 running_ 被置为 false。
// 每个周期执行以下步骤：
//   1. 维护连接（断线重连，指数退避 + 随机抖动）；
//   2. 按独立周期发布流状态信息（stream_info），方便云端实时掌握设备流配置；
//   3. 按独立周期上报健康心跳（health_report），告知工控机节点存活；
//   4. 批量拉取待上传记录并逐条发布；
//   5. 根据发布结果更新 SQLite 中的 uploaded / failed / dead 状态；
//   6. 使用条件变量等待下一次轮询或新数据通知，减少固定轮询延迟。
//
// 线程管理：由 start()/stop() 控制生命周期。
// ============================================================================
void MqttUploader::uploadLoop() {
    utils::applyThreadRuntime("mqtt_upload", "mqtt-upload"); // 应用实时线程调优（绑核、优先级）

    if (!store_) {                             // 无有效存储
        return;
    }

    int interval_ms = config_.publish_interval_sec * 1000; // 轮询间隔（ms）
    if (interval_ms <= 0) {
        interval_ms = 60000;                   // 默认 60s
    }

    int retry_interval_ms = config_.retry_interval_sec * 1000; // 失败重试间隔（ms）
    if (retry_interval_ms <= 0) {
        retry_interval_ms = 30000;             // 默认 30s
    }

    int reconnect_backoff_ms = config_.reconnect_backoff_min_ms; // 重连退避初始值

    start_ts_ms_ = nowMs();                    // 记录启动时间戳（用于健康心跳 uptime 计算）
    std::int64_t last_stream_info_ms = 0;      // 上次流信息上报时间

    while (running_) {                         // 主循环：running_ 为 false 时退出
        const std::int64_t now_ms = nowMs();   // 获取当前时间戳

        // ----- 阶段 1：连接维护（断线重连） -----
        if (!connected_.load()) {              // 未连接
            reconnect_attempts_.fetch_add(1, std::memory_order_relaxed); // 原子递增重连尝试计数
            connectBroker();                   // 尝试建立连接
            if (connected_.load()) {           // 连接成功
                reconnect_backoff_ms = config_.reconnect_backoff_min_ms; // 重置退避窗口
                continue;                      // 立即进入下一轮，不用等待
            }

            // 计算重连等待时间 = 退避基数 + 随机抖动
            int jitter_ms = 0;                 // 随机抖动值（ms）
            if (config_.reconnect_jitter_ms > 0) {
                thread_local std::mt19937 rng{std::random_device{}()};             // 线程局部 Mersenne Twister 生成器
                std::uniform_int_distribution<int> dist(0, config_.reconnect_jitter_ms); // [0, jitter] 均匀分布
                jitter_ms = dist(rng);         // 生成随机抖动
            }
            const int wait_ms = reconnect_backoff_ms + jitter_ms; // 最终等待时长

            std::unique_lock<std::mutex> lk(upload_mtx_);        // 持有锁
            upload_cv_.wait_for(lk, std::chrono::milliseconds(wait_ms), // 条件变量等待（可被 notifyNewData 提前唤醒）
                                [this]() { return !running_.load() || has_new_data_; });
            has_new_data_ = false;             // 消费新数据标志

            // 指数退避：每次失败等待时间翻倍，直到达到上限
            reconnect_backoff_ms = std::min(
                reconnect_backoff_ms * 2,      // 翻倍
                std::max(config_.reconnect_backoff_min_ms, config_.reconnect_backoff_max_ms)); // 取上限
            continue;                          // 回到 while 循环顶部重新判断
        }

        // 连接正常时重置退避基数
        reconnect_backoff_ms = config_.reconnect_backoff_min_ms;

        // ----- 阶段 2：流状态信息定期上报 -----
        if (config_.stream_info_enable) {      // 启用了流信息上报
            std::int64_t info_interval_ms = config_.stream_info_interval_sec * 1000; // 上报间隔
            if (info_interval_ms <= 0) {
                info_interval_ms = 60000;      // 默认 60s
            }

            // 检查是否达到上报时间点
            if (last_stream_info_ms == 0 || (now_ms - last_stream_info_ms) >= info_interval_ms) {
                network::protocol::DataEnvelope env;  // 创建数据信封
                env.version = 1;                      // 协议版本
                env.type = "stream_info";             // 数据类型
                env.ts_ms = now_ms;                   // 当前时间戳
                env.device_id = config_.device_id;    // 设备ID
                env.site_id = config_.site_id;        // 站点ID
                env.fw_version = config_.fw_version;  // 固件版本
                env.payload_json = buildStreamInfoPayload(config_.streams); // 流信息 JSON

                const std::string json = env.toJson();                     // 序列化为 JSON
                const std::string topic = config_.stream_info_topic.empty()
                    ? config_.topic : config_.stream_info_topic;            // 使用配置或默认主题
                if (publishPayloadToTopic(topic, json)) {                   // 发布
                    last_stream_info_ms = now_ms;                          // 更新上报时间
                }
            }
        }

        // ----- 阶段 3：健康心跳上报 -----
        if (config_.health_report_enable) {    // 启用了健康心跳
            const int64_t health_interval_ms = config_.health_report_interval_sec * 1000LL; // 心跳间隔
            if (health_interval_ms > 0 && (last_health_report_ms_ == 0
                || (now_ms - last_health_report_ms_) >= health_interval_ms)) {  // 到时间了
                std::ostringstream hb;         // 构建心跳 JSON
                hb << "{\"type\":\"health_report\""
                   << ",\"device_id\":\"" << network::protocol::escapeJson(config_.device_id) << "\""
                   << ",\"site_id\":\"" << network::protocol::escapeJson(config_.site_id) << "\""
                   << ",\"fw_version\":\"" << network::protocol::escapeJson(config_.fw_version) << "\""
                   << ",\"ts_ms\":" << now_ms                          // 当前时间戳
                   << ",\"uptime_ms\":" << (now_ms - start_ts_ms_)     // 运行时长（ms）
                   << ",\"mqtt_connected\":" << (connected_.load() ? "true" : "false") // 连接状态
                   << "}";
                const std::string topic = config_.health_report_topic.empty()
                    ? config_.topic + "/health" : config_.health_report_topic; // 心跳主题
                if (publishPayloadToTopic(topic, hb.str())) {                   // 发布心跳
                    last_health_report_ms_ = now_ms;                            // 更新上报时间
                }
            }
        }

        // ----- 阶段 4：批量拉取待上传记录 -----
        std::vector<data_lifecycle::UploadRecord> records; // 待上传记录容器
        int limit = store_->maxBatch();                    // 数据库单次 fetch 上限
        if (limit <= 0) {
            limit = 200;                                   // 默认上限 200 条
        }

        // 批量抓取待上传数据，减少数据库 I/O 次数，提高吞吐
        if (store_->fetchPending(limit, records)) {        // 从 SQLite 拉取待处理记录
            std::vector<std::int64_t> uploaded_ids;        // 上传成功记录 ID 集合
            std::vector<std::int64_t> failed_ids;          // 上传失败记录 ID 集合
            std::vector<std::int64_t> dead_ids;            // 死亡（超过最大重试）记录 ID 集合

            for (const auto& rec : records) {              // 逐条处理
                // 连接丢失时中止本轮发送，进入下一轮重连逻辑
                if (!connected_.load()) {
                    break;                                 // 跳出循环
                }

                // 超过最大重试次数的记录直接标记 dead，避免无限重试占用资源
                if (config_.max_retry > 0 && rec.retry_count >= config_.max_retry) {
                    dead_ids.push_back(rec.id);            // 加入死亡列表
                    continue;                              // 跳过本条
                }

                // 退避窗口内的失败记录先跳过，等待下一轮再尝试
                if (rec.last_try_ts > 0 && (now_ms - rec.last_try_ts) < retry_interval_ms) {
                    continue;                              // 跳过，等待冷却
                }

                // 构建 DataEnvelope 封装
                network::protocol::DataEnvelope envelope;  // 数据信封
                envelope.version = 1;                      // 协议版本
                envelope.type = rec.type;                  // 数据类型（来自数据库记录）
                envelope.ts_ms = rec.ts_ms;                // 原始时间戳
                envelope.payload_json = rec.payload;       // JSON 负载
                envelope.device_id = config_.device_id;    // 设备ID
                envelope.site_id = config_.site_id;        // 站点ID
                envelope.fw_version = config_.fw_version;  // 固件版本

                const std::string json = envelope.toJson(); // 序列化
                if (publishPayload(json)) {                 // 发布到默认主题
                    uploaded_ids.push_back(rec.id);         // 成功：加入成功列表
                } else {
                    // 连接中断不计入业务失败重试额度，避免无效消耗 retry_count
                    if (!connected_.load()) {
                        break;                             // 连接中断，中止本轮
                    }
                    failed_ids.push_back(rec.id);           // 加入失败列表
                }
            }

            // ----- 阶段 5：批量回写数据库状态 -----
            // 分别批量更新，减少单条 SQL 带来的 I/O 开销
            if (!uploaded_ids.empty()) {
                store_->markUploaded(uploaded_ids);         // 标记为已上传
            }
            if (!failed_ids.empty()) {
                store_->markFailed(failed_ids, "publish_failed", now_ms); // 标记失败（递增 retry_count）
            }
            if (!dead_ids.empty()) {
                store_->markDead(dead_ids, "max_retry");    // 标记死亡（不再重试）
            }
        }

        // ----- 阶段 6：条件变量等待（替代固定 sleep） -----
        // 两个唤醒条件：
        //   1. 到达常规间隔后自动唤醒，执行下一轮轮询；
        //   2. 收到 notifyNewData() 时提前唤醒，降低新数据上报延迟。
        {
            std::unique_lock<std::mutex> lk(upload_mtx_);
            upload_cv_.wait_for(lk, std::chrono::milliseconds(interval_ms), // 等待 interval_ms 或被唤醒
                                [this]() { return !running_.load() || has_new_data_; });
            has_new_data_ = false;             // 消费"有新数据"标志，避免重复立即唤醒
        }
    }
}

// ============================================================================
// 获取上传统计快照（原子读取各计数器）
// @return  UploadStats 包含重连次数、发布成功/失败计数、流信息成功/失败计数
// ============================================================================
MqttUploader::UploadStats MqttUploader::statsSnapshot() const {
    UploadStats stats;                                                       // 栈上构造
    stats.reconnect_attempts = reconnect_attempts_.load(std::memory_order_relaxed);   // 原子读
    stats.publish_success = publish_success_.load(std::memory_order_relaxed);         // 原子读
    stats.publish_failed = publish_failed_.load(std::memory_order_relaxed);           // 原子读
    stats.stream_info_success = stream_info_success_.load(std::memory_order_relaxed); // 原子读
    stats.stream_info_failed = stream_info_failed_.load(std::memory_order_relaxed);   // 原子读
    return stats;                                                            // 返回快照副本
}

// ============================================================================
// 获取当前系统时间戳（毫秒），基于 system_clock（挂钟时间）。
// @return  自 Unix 纪元以来的毫秒数
// ============================================================================
std::int64_t MqttUploader::nowMs() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(  // 转换为毫秒精度
        std::chrono::system_clock::now().time_since_epoch()).count(); // 获取当前时刻
}

// ============================================================================
// 连接事件处理（由 mosquitto onConnect 回调调用）：
//   1. 更新 connected_ 原子变量；
//   2. 连接成功后发布 online 状态消息，告知云端设备已上线。
//
// @param rc  连接结果码，0 表示成功，非 0 表示失败
// ============================================================================
void MqttUploader::handleConnectEvent(int rc) {
#ifdef HAVE_MOSQUITTO
    const bool ok = (rc == 0);              // rc==0 → 连接成功
    connected_.store(ok);                   // 原子更新连接状态

    if (!ok || !client_) {                  // 连接失败或无客户端
        return;
    }

    // 连接成功后即发布 online 状态（遗嘱消息已在初始化时设置 offline）
    const std::string status_topic = config_.health_report_topic.empty()
        ? config_.topic + "/status" : config_.health_report_topic; // 状态主题
    const std::string online_payload = "{\"device_id\":\"" + config_.device_id +
                                       "\",\"online\":true}";        // 在线消息内容
    mosquitto_publish(static_cast<mosquitto*>(client_), nullptr, status_topic.c_str(),
                      static_cast<int>(online_payload.size()), online_payload.c_str(), 1, true); // QoS=1, retain=true
#else
    (void)rc;                               // 消除未使用变量警告
#endif
}

// ============================================================================
// 断连事件处理（由 mosquitto onDisconnect 回调调用）：
// 仅更新连接状态为 false，重连逻辑由 uploadLoop 中的 connectBroker 驱动。
//
// @param rc  断连原因码（mosquitto 定义）
// ============================================================================
void MqttUploader::handleDisconnectEvent(int) {
    connected_.store(false);                // 原子标记离线
}

// ============================================================================
// 外部数据写入后调用该接口，通知上传线程尽快执行下一轮发送。
// 使用条件变量实现：设置 has_new_data_ 标志并 notify_one，
// 使 uploadLoop 从 wait_for 中提前醒来。
// ============================================================================
void MqttUploader::notifyNewData() {
    {
        std::lock_guard<std::mutex> lk(upload_mtx_); // 加锁
        has_new_data_ = true;                         // 置位新数据标志
    }
    upload_cv_.notify_one();                          // 唤醒一个等待线程
}

} // namespace network
