// =============================================================================
// app_initializer.cpp —— 应用初始化器实现
// 按依赖关系串联所有子系统：存储 -> MQTT -> 分发器 -> 推理服务 -> FrameHub -> 视频管线
// 初始化顺序绝对不能错乱：线程RT配置必须在创建线程之前，推理服务必须在管线之前
// =============================================================================

#include "app/app_initializer.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "config/ini_config.h"
#include "utils/thread_runtime.h"

namespace {

// =============================================================================
// 单次启动管线。重试策略与具体启动动作分离，调用处可以直接看到启动入口。
// =============================================================================
template<typename Pipeline>
bool pipeline_start(Pipeline& pipeline) {
    return pipeline->start();
}

// 首次启动失败后按指数退避重试：1s, 2s, 4s。
template<typename Pipeline>
bool retry(Pipeline& pipeline, const char* name,
           int max_retries = 3, int base_delay_ms = 1000) {
    for (int attempt = 1; attempt <= max_retries; ++attempt) {
        const int delay = base_delay_ms * (1 << (attempt - 1));
        std::cerr << "[" << name << "] retry " << attempt << '/' << max_retries
                  << " after " << delay << "ms\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));

        if (pipeline_start(pipeline)) {
            std::cout << "[" << name << "] started successfully on retry "
                      << attempt << '\n';
            return true;
        }
    }
    std::cerr << "[" << name << "] failed to start after " << max_retries
              << " retries\n";
    return false;
}

// 视频流 ID 常量：用于数据库记录区分及推理回调路由
constexpr int STREAM_ID_EXTERNAL_RTSP = 1;
constexpr int STREAM_ID_IMX415 = 2;

// =============================================================================
// parseModelMask —— 解析逗号分隔的模型名称为位掩码
// 三模型架构：coco(bit0), fire(bit1), ppe(bit2)
// 大小写不敏感，自动去除空白，向后兼容 "person"/"cat"/"dog" 旧名称
// =============================================================================
uint8_t parseModelMask(const std::string& value) {
    uint8_t mask = 0;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }), token.end());
        std::transform(token.begin(), token.end(), token.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });

        if (token == "coco") mask |= 1u << 0;
        else if (token == "fire") mask |= 1u << 1;
        else if (token == "ppe") mask |= 1u << 2;
        else if (token == "person") mask |= 1u << 0;   // 向后兼容
        else if (token == "cat") mask |= 1u << 0;
        else if (token == "dog") mask |= 1u << 0;
        else if (!token.empty()) std::cerr << "[InferenceRoute] unknown model: " << token << '\n';
    }
    return mask;
}

// =============================================================================
// publishDmaFrameToHub —— DMA帧发布至FrameHub（多路汇聚链路的核心粘合函数）
// =============================================================================
void publishDmaFrameToHub(AppContext& context,
                          pipeline::FrameSource source,
                          const pipeline::FrameHub::DmaFrame& frame,
                          std::chrono::system_clock::time_point ts,
                          int64_t capture_mono_ns,
                          const pipeline::FrameHub::FrameOverlay& overlay) {
    context.frame_hub->updateDma(source, frame, ts, capture_mono_ns, overlay);
}

// =============================================================================
// loadConfigPath —— 配置路径解析：环境变量 EDGE_SENSOR_CONFIG 优先，回退默认路径
// =============================================================================
std::string loadConfigPath() {
    const char* cfg_env = std::getenv("EDGE_SENSOR_CONFIG");
    return cfg_env ? cfg_env : "../config/sensors.ini";
}

// =============================================================================
// readOneLine —— 读取文本文件第一行（用于读取 sysfs/procfs 单行值）
// =============================================================================
std::string readOneLine(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";
    std::string line;
    std::getline(file, line);
    return line;
}

// =============================================================================
// parseCpuList —— 解析 CPU 列表字符串（如 "0-3,6,8-11"），去重排序后返回
// =============================================================================
std::vector<int> parseCpuList(const std::string& cpus) {
    std::vector<int> out;
    std::stringstream ss(cpus);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) {
            return std::isspace(c) != 0;
        }), token.end());
        if (token.empty()) continue;

        const auto dash = token.find('-');
        if (dash == std::string::npos) {
            try { out.push_back(std::stoi(token)); } catch (...) {}
            continue;
        }

        try {
            int begin = std::stoi(token.substr(0, dash));
            int end = std::stoi(token.substr(dash + 1));
            if (begin > end) std::swap(begin, end);
            for (int cpu = begin; cpu <= end; ++cpu) out.push_back(cpu);
        } catch (...) {}
    }

    // 去重且升序：保证后续子集判断正确
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// =============================================================================
// isSubset —— 判断 child 是否为 parent 的子集（前提：两者已排序）
// =============================================================================
bool isSubset(const std::vector<int>& child, const std::vector<int>& parent) {
    return std::includes(parent.begin(), parent.end(), child.begin(), child.end());
}

// =============================================================================
// detectIsolatedCpus —— 探测内核隔离 CPU 列表
// 方式一：sysfs /sys/devices/system/cpu/isolated（较新内核）
// 方式二：解析 /proc/cmdline 中 isolcpus= 参数（回退方案）
// =============================================================================
std::string detectIsolatedCpus() {
    std::string isolated = readOneLine("/sys/devices/system/cpu/isolated");
    if (!isolated.empty()) return isolated;

    const std::string cmdline = readOneLine("/proc/cmdline");
    const std::string key = "isolcpus=";
    const auto pos = cmdline.find(key);
    if (pos == std::string::npos) return "";

    const auto begin = pos + key.size();
    const auto end = cmdline.find(' ', begin);
    if (end == std::string::npos) return cmdline.substr(begin);
    return cmdline.substr(begin, end - begin);
}

// =============================================================================
// validateThreadRtEnvironment —— 校验 PREEMPT_RT 环境与 CPU 隔离配置
// 检查项：1) PREEMPT_RT 内核标志  2) isolcpus 是否设置  3) 进程CPU范围是否在隔离CPU内
// =============================================================================
void validateThreadRtEnvironment(const std::string& cfg_path) {
    IniConfig cfg;
    if (!cfg.load(cfg_path)) return;

    if (!cfg.getBool("thread_rt", "enable", false)) return;

    const bool require_preempt_rt = cfg.getBool("thread_rt", "require_preempt_rt", false);
    const std::string process_cpus = cfg.getString(
        "thread_rt", "process_cpus", cfg.getString("thread_rt", "cpus", ""));

    const std::string realtime_flag = readOneLine("/sys/kernel/realtime");
    const bool preempt_rt = (!realtime_flag.empty() && realtime_flag[0] == '1');
    const std::string isolated_cpus = detectIsolatedCpus();

    bool pass = true;
    if (require_preempt_rt && !preempt_rt) {
        pass = false;
        std::cerr << "[RtCheck] WARN require_preempt_rt=true but /sys/kernel/realtime!=1" << std::endl;
    }

    if (isolated_cpus.empty()) {
        pass = false;
        std::cerr << "[RtCheck] WARN isolated cpu list not detected (isolcpus/nohz_full not set)" << std::endl;
    } else if (!process_cpus.empty()) {
        const auto process_set = parseCpuList(process_cpus);
        const auto isolated_set = parseCpuList(isolated_cpus);
        if (!process_set.empty() && !isolated_set.empty() && !isSubset(process_set, isolated_set)) {
            pass = false;
            std::cerr << "[RtCheck] WARN process_cpus=" << process_cpus
                      << " not fully in isolated cpus=" << isolated_cpus << std::endl;
        }
    }

    std::cout << "[RtCheck] status=" << (pass ? "PASS" : "WARN")
              << " preempt_rt=" << preempt_rt
              << " require_preempt_rt=" << require_preempt_rt
              << " process_cpus=" << process_cpus
              << " isolated_cpus=" << (isolated_cpus.empty() ? "<none>" : isolated_cpus)
              << std::endl;
}

// =============================================================================
// validateDefaultMosaicTopology —— 确认 Mosaic 所需的所有视频源管线均已启动
// =============================================================================
void validateDefaultMosaicTopology(const AppContext& context) {
    if (!context.mosaic_pipeline) {
        std::cerr << "[TopologyCheck] mosaic pipeline is not running" << std::endl;
        return;
    }

    std::vector<const char*> missing_sources;
    if (!context.external_rtsp_pipeline) missing_sources.push_back("external_rtsp");
    if (context.imx415_cfg.enable && !context.imx415_pipeline) missing_sources.push_back("imx415");

    if (missing_sources.empty()) {
        const int expected_sources = context.imx415_cfg.enable ? 2 : 1;
        std::cout << "[TopologyCheck] mosaic default topology ready: "
                  << expected_sources << " source pipeline(s) are running" << std::endl;
        return;
    }

    std::cerr << "[TopologyCheck] upstream video pipelines are missing:";
    for (const char* name : missing_sources) std::cerr << ' ' << name;
    std::cerr << std::endl;
}

// =============================================================================
// createInferenceWriter —— 推理结果写入器工厂（闭包链）
// 内层闭包完成两件事：
//   1. 将检测事件推送给 Mosaic 拼接管线（触发事件联动录制）
//   2. 将推理统计提交给异步分发器落库（或兜底同步直写）
// =============================================================================
template<typename Context>
auto createInferenceWriter(Context& context) {
    return [&context](int stream_id) {
        return [&context, stream_id](const data_lifecycle::InferenceStats& stats) {
            // ---- Mosaic 检测事件推送（触发 OSD 叠加 + 事件录制） ----
            if (context.mosaic_pipeline) {
                const int64_t mono_ns = stats.capture_mono_ns > 0 ? stats.capture_mono_ns :
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                context.mosaic_pipeline->updateDetection(recording::EventType::Person,
                    stats.person_count > 0, mono_ns, stats.ts_ms, stream_id);
                context.mosaic_pipeline->updateDetection(recording::EventType::Cat,
                    stats.cat_count > 0, mono_ns, stats.ts_ms, stream_id);
                context.mosaic_pipeline->updateDetection(recording::EventType::Dog,
                    stats.dog_count > 0, mono_ns, stats.ts_ms, stream_id);
                context.mosaic_pipeline->updateDetection(recording::EventType::Fire,
                    stats.fire_count > 0, mono_ns, stats.ts_ms, stream_id);
                context.mosaic_pipeline->updateDetection(recording::EventType::Smoke,
                    stats.smoke_count > 0, mono_ns, stats.ts_ms, stream_id);
                context.mosaic_pipeline->updateDetection(recording::EventType::Ppe,
                    stats.ppe_count > 0, mono_ns, stats.ts_ms, stream_id);
            }

            // ---- 推理结果落库 ----
            if (!context.storage_enabled) return;

            if (context.inference_dispatcher) {
                // 正常路径：异步分发（带采样节流）
                context.inference_dispatcher->submit(stream_id, stats);
                return;
            }

            // 兜底路径：分发器未启动时直接同步写库（无节流）
            context.sqlite_store.insertInferenceStats(stats);
            if (context.mqtt_enabled) context.mqtt_uploader.notifyNewData();
        };
    };
}

} // namespace

// =============================================================================
// AppInitializer::initialize —— 按依赖顺序串联所有模块
// 初始化顺序：
//   0. 线程RT配置 + NPU温控配置（必须在任何线程创建前完成）
//   1. 基础服务：存储 -> MQTT -> 推理参数
//   2. 推理分发器（依赖存储可用）
//   3. 全局共享推理服务（NPU模型加载，耗时最长）
//   4. FrameHub 帧汇聚中心
//   5. 各视频管线（依赖推理服务和FrameHub）
//   6. 拓扑完整性校验
//   7. 启动NPU温控监控线程
// =============================================================================
bool AppInitializer::initialize(AppContext& context) {
    const std::string cfg_path = loadConfigPath();

    // ---- 步骤 0：线程实时调度（必须在创建任何线程之前） ----
    utils::configureThreadRuntimeFromIni(cfg_path);
    utils::applyProcessRuntime("rk3568-main");
    validateThreadRtEnvironment(cfg_path);

    // ---- 步骤 0.1：NPU 温控初始化 ----
    utils::ThermalConfig thermal_cfg = context.thermal_manager.config();
    utils::NpuThermalManager::loadFromIni(cfg_path, thermal_cfg);
    context.thermal_manager.configure(thermal_cfg);

    // ---- 步骤 0.2：N12优化——启动时输出配置摘要便于运维排查 ----
    printConfigSummary(cfg_path);

    // ---- 步骤 1：基础服务（存储 -> MQTT -> 推理参数） ----
    initStorage(context, cfg_path);
    initMqtt(context, cfg_path);
    context.inference_sample_ms = context.storage_cfg.inference_sample_ms;

    // ---- 步骤 2：推理结果异步分发器（依赖：存储已就绪） ----
    if (context.storage_enabled) {
        app::InferenceDispatchPolicy policy;
        policy.inference_sample_ms = context.inference_sample_ms;

        // 从 INI [inference_dispatch] 段覆盖策略参数
        IniConfig infer_dispatch_cfg;
        if (infer_dispatch_cfg.load(cfg_path)) {
            policy.queue_size = infer_dispatch_cfg.getInt("inference_dispatch", "queue_size", policy.queue_size);
            policy.warn_on_drop = infer_dispatch_cfg.getBool("inference_dispatch", "warn_on_drop", policy.warn_on_drop);
            policy.drop_warn_interval_ms = infer_dispatch_cfg.getInt(
                "inference_dispatch", "drop_warn_interval_ms", policy.drop_warn_interval_ms);
            policy.dynamic_sample_enable = infer_dispatch_cfg.getBool(
                "inference_dispatch", "dynamic_sample_enable", policy.dynamic_sample_enable);
            policy.dynamic_sample_step_ms = infer_dispatch_cfg.getInt(
                "inference_dispatch", "dynamic_sample_step_ms", policy.dynamic_sample_step_ms);
            policy.dynamic_sample_max_ms = infer_dispatch_cfg.getInt(
                "inference_dispatch", "dynamic_sample_max_ms", policy.dynamic_sample_max_ms);
        }

        context.inference_dispatcher = std::make_unique<app::InferenceResultDispatcher>();
        if (!context.inference_dispatcher->start(
                &context.sqlite_store,
                policy,
                [&context]() {
                    if (context.mqtt_enabled) context.mqtt_uploader.notifyNewData();
                })) {
            context.inference_dispatcher.reset();
        }
    }

    // ---- 步骤 3：全局共享推理服务（加载三模型 RKNN 到 NPU） ----
    initInferenceService(context, cfg_path);

    // ---- 步骤 4：帧汇聚中心（多管线帧的中转站） ----
    context.frame_hub = std::make_shared<pipeline::FrameHub>();

    // ---- 步骤 5：初始化各视频管线 ----
    auto inference_writer = createInferenceWriter(context);
    initVideoPipelines(context, cfg_path, inference_writer);

    // ---- 步骤 6：拓扑完整性自检 ----
    validateDefaultMosaicTopology(context);

    // ---- 步骤 7：启动 NPU 温控监控（从 sysfs 读取温度） ----
    if (!context.thermal_manager.start()) {
        std::cerr << "[Thermal] monitor not started, fallback to non-thermal-aware runtime" << std::endl;
    }

    return true;
}

// =============================================================================
// AppInitializer::shutdown —— 按逆序优雅释放所有资源
// 关闭顺序与初始化严格相反：
//   先停下游（管线） -> 推理服务（管线依赖） -> 分发器 -> 温控 -> MQTT -> 存储
// =============================================================================
void AppInitializer::shutdown(AppContext& context) {
    // 先停最下游，确保不再有RGA任务读取采集DMA缓冲。
    if (context.mosaic_pipeline) context.mosaic_pipeline->stop();
    // 再停生产者；V4L2租约状态会先失效，残留快照随后clear时不会对关闭FD执行QBUF。
    if (context.external_rtsp_pipeline) context.external_rtsp_pipeline->stop();
    if (context.imx415_pipeline) context.imx415_pipeline->stop();
    if (context.frame_hub) context.frame_hub->clear();

    // 共享推理服务必须在所有管线 stop 之后释放（管线推理线程依赖此服务）
    if (context.inference_service) {
        context.inference_service->shutdown();
        context.inference_service.reset();
    }

    // 停止分发器：确保队列中的统计已全部落库
    if (context.inference_dispatcher) {
        context.inference_dispatcher->stop();
        context.inference_dispatcher.reset();
    }

    context.thermal_manager.stop();

    // MQTT 关闭：断开 broker 连接
    if (context.mqtt_enabled) context.mqtt_uploader.stop();

    // 数据库最后关闭（MQTT 上传器依赖数据库读取待上报数据）
    context.sqlite_store.close();
}

// =============================================================================
// AppInitializer::printConfigSummary —— N12优化：启动时输出配置摘要
// =============================================================================
void AppInitializer::printConfigSummary(const std::string& cfg_path) {
    IniConfig cfg;
    if (!cfg.load(cfg_path)) {
        std::cerr << "[ConfigSummary] Failed to load config: " << cfg_path << std::endl;
        return;
    }

    std::cout << "\n========== Config Summary (effective) ==========" << std::endl;
    std::cout << "  Config file : " << cfg_path << std::endl;

    std::cout << "  [external_rtsp] enable=" << cfg.getBool("video_source_external", "enable", false)
              << "  url=" << cfg.getString("video_source_external", "url", "-") << std::endl;
    std::cout << "  [imx415]        enable=" << cfg.getBool("video_source_imx415", "enable", false)
              << "  device=" << cfg.getString("video_source_imx415", "device", "-") << std::endl;

    std::cout << "  [inference]     visible=" << cfg.getBool("visible_inference", "enable", false)
              << "  external_models=" << cfg.getString("visible_inference", "external_models", "person")
              << "  imx415_models=" << cfg.getString("visible_inference", "imx415_models", "cat,dog")
              << std::endl;

    std::cout << "  [thermal]       enable=" << cfg.getBool("thermal", "enable", true)
              << "  light=" << cfg.getInt("thermal", "threshold_light", 65)
              << "  medium=" << cfg.getInt("thermal", "threshold_medium", 75)
              << "  heavy=" << cfg.getInt("thermal", "threshold_heavy", 85)
              << "  poll_ms=" << cfg.getInt("thermal", "poll_interval_ms", 2000)
              << std::endl;

    std::cout << "  [inference_dispatch] queue=" << cfg.getInt("inference_dispatch", "queue_size", 256)
              << "  dynamic_sample=" << cfg.getBool("inference_dispatch", "dynamic_sample_enable", true)
              << "  step_ms=" << cfg.getInt("inference_dispatch", "dynamic_sample_step_ms", 1000)
              << "  max_ms=" << cfg.getInt("inference_dispatch", "dynamic_sample_max_ms", 15000)
              << std::endl;

    std::cout << "  [storage]       enable=" << cfg.getBool("storage", "enable", false)
              << "  wal=" << cfg.getBool("storage", "wal", false)
              << "  batch=" << cfg.getInt("storage", "write_batch_size", 0) << std::endl;

    bool tls_insecure = cfg.getBool("mqtt", "tls_insecure", false);
    std::cout << "  [mqtt]          enable=" << cfg.getBool("mqtt", "enable", false)
              << "  tls=" << cfg.getBool("mqtt", "tls_enable", false)
              << "  tls_insecure=" << tls_insecure << std::endl;
    if (tls_insecure) {
        std::cerr << "  ⚠ WARNING: mqtt.tls_insecure=true, vulnerable to MITM attacks!" << std::endl;
    }

    std::cout << "  [mosaic]        enable=" << cfg.getBool("mosaic_stream", "enable", false)
              << "  layout=" << cfg.getString("mosaic_stream", "input_mode", "side_by_side")
              << "  output_url=" << cfg.getString("stream_output", "url", "") << std::endl;
    std::cout << "=================================================\n" << std::endl;
}

// =============================================================================
// initStorage —— 初始化 SQLite 存储
// 加载 [storage] 段配置，开启 WAL 模式与批量写入
// =============================================================================
void AppInitializer::initStorage(AppContext& context, const std::string& cfg_path) {
    context.storage_enabled = false;
    if (!data_lifecycle::SqliteStore::loadFromIni(cfg_path, context.storage_cfg)) return;
    if (!context.storage_cfg.enable) return;

    if (context.sqlite_store.initialize(context.storage_cfg)) {
        context.storage_enabled = true;
    } else {
        std::cerr << "SQLite store init failed" << std::endl;
    }
}

// =============================================================================
// initMqtt —— 初始化 MQTT 上传器
// 依赖存储可用，因为 MQTT 从 SQLite 读取待上报数据
// =============================================================================
void AppInitializer::initMqtt(AppContext& context, const std::string& cfg_path) {
    context.mqtt_enabled = false;
    if (!network::MqttUploader::loadFromIni(cfg_path, context.mqtt_cfg)) return;
    if (!context.mqtt_cfg.enable) return;
    if (!context.storage_enabled) return;

    if (context.mqtt_uploader.initialize(context.mqtt_cfg, &context.sqlite_store)) {
        context.mqtt_enabled = true;
        context.mqtt_uploader.start();
    } else {
        std::cerr << "MQTT uploader init failed" << std::endl;
    }
}

// =============================================================================
// initInferenceService —— 初始化全局共享推理服务
// 加载三模型架构（COCO / FireSmoke / PPE）的 RKNN 文件到 NPU
// 通过 source_model_masks 控制每路视频使用哪些模型
// stream_class_filter 按流过滤检测类别：A路只关注person+PPE，B路关注猫狗+火焰
// 将温控管理器注入推理服务，实现"高温自动跳帧"的闭环热保护
// =============================================================================
void AppInitializer::initInferenceService(AppContext& context, const std::string& cfg_path) {
    IniConfig cfg;
    if (!cfg.load(cfg_path)) return;

    const bool vis_infer = cfg.getBool("visible_inference", "enable", true);
    if (!vis_infer) return;

    pipeline::InferenceServiceConfig svc_cfg;

    // Model 0: YOLOv8n-COCO (80类) — A路人员检测 + B路猫狗检测
    svc_cfg.models[0] = {
        "coco",
        cfg.getString("visible_inference", "model_coco", "../model/yolov8n_coco_int8.rknn"),
        cfg.getString("visible_inference", "label_coco", "../model/labels/coco_trimmed.txt"),
        80,
        {{ std::set<int>{0}, std::set<int>{15, 16} }}  // A路:person(0), B路:cat(15)/dog(16)
    };

    // Model 1: YOLOv8n Fire/Smoke (2类) — 二分类轻量模型
    svc_cfg.models[1] = {
        "fire",
        cfg.getString("visible_inference", "model_fire", "../model/yolov8n_fire_smoke_int8.rknn"),
        cfg.getString("visible_inference", "label_fire", "../model/labels/fire_smoke_labels.txt"),
        2,
        {{ std::set<int>{}, std::set<int>{} }}  // 全部保留
    };

    // Model 2: YOLOv8n SH17-PPE (17类→过滤为2类) — A路PPE检测
    svc_cfg.models[2] = {
        "ppe",
        cfg.getString("visible_inference", "model_ppe", "../model/yolov8n_sh17_ppe_int8.rknn"),
        cfg.getString("visible_inference", "label_ppe", "../model/labels/sh17_ppe_trimmed.txt"),
        17,
        {{ std::set<int>{10, 15}, std::set<int>{} }}  // A路:helmet(10)/safety-suit(15), B路不启用
    };

    // 后处理阈值：confidence_threshold 越低检出越多但误报增加
    svc_cfg.models[0].confidence_threshold = static_cast<float>(
        cfg.getDouble("visible_inference", "coco_confidence", 0.45));
    svc_cfg.models[0].nms_threshold = static_cast<float>(
        cfg.getDouble("visible_inference", "coco_nms_iou", 0.45));
    svc_cfg.models[1].confidence_threshold = static_cast<float>(
        cfg.getDouble("visible_inference", "fire_confidence", 0.40));
    svc_cfg.models[1].nms_threshold = static_cast<float>(
        cfg.getDouble("visible_inference", "fire_nms_iou", 0.45));
    svc_cfg.models[2].confidence_threshold = static_cast<float>(
        cfg.getDouble("visible_inference", "ppe_confidence", 0.40));
    svc_cfg.models[2].nms_threshold = static_cast<float>(
        cfg.getDouble("visible_inference", "ppe_nms_iou", 0.45));

    // 每路视频的模型位掩码：控制该路使用哪些模型
    svc_cfg.source_model_masks[0] = parseModelMask(
        cfg.getString("visible_inference", "external_models", "coco,ppe"));
    svc_cfg.source_model_masks[1] = parseModelMask(
        cfg.getString("visible_inference", "imx415_models", "coco,fire"));

    // 每帧最多运行的模型数：避免单帧过度占用 NPU
    svc_cfg.source_models_per_frame[0] = std::max(1,
        cfg.getInt("visible_inference", "external_models_per_frame", 1));
    svc_cfg.source_models_per_frame[1] = std::max(1,
        cfg.getInt("visible_inference", "imx415_models_per_frame", 1));

    context.inference_service = std::make_shared<pipeline::InferenceService>();
    if (!context.inference_service->initialize(svc_cfg)) {
        std::cerr << "InferenceService init failed" << std::endl;
        context.inference_service.reset();
    } else {
        // 关键：将温控管理器注入推理服务，实现高温降载保护闭环
        context.inference_service->setThermalManager(&context.thermal_manager);
    }
}

// =============================================================================
// initVideoPipelines —— 按依赖顺序初始化所有视频管线
// 顺序：外部 RTSP -> IMX415 -> Mosaic 拼接（Mosaic 通过 FrameHub 消费前两者输出）
// =============================================================================
template<typename InferenceWriter>
void AppInitializer::initVideoPipelines(AppContext& context, const std::string& cfg_path,
                                         InferenceWriter inference_writer) {
    initExternalRtspPipeline(context, cfg_path, inference_writer);
    initImx415Pipeline(context, cfg_path, inference_writer);
    initMosaicPipeline(context, cfg_path);
}

// =============================================================================
// initExternalRtspPipeline —— 初始化外部 RTSP 拉流管线（A路）
// 流程：加载配置 -> 实例化管线 -> 注入推理服务 -> 绑定推理回调/帧推送回调 -> 启动
// =============================================================================
template<typename InferenceWriter>
void AppInitializer::initExternalRtspPipeline(AppContext& context, const std::string& cfg_path,
                                               InferenceWriter inference_writer) {
    pipeline::VisibleRtspPipelineConfig external_cfg;
    if (!pipeline::VisibleRtspPipeline::loadFromIni(cfg_path, external_cfg, "video_source_external")
        || !external_cfg.enable) return;

    context.external_rtsp_cfg = external_cfg;
    context.external_rtsp_pipeline = std::make_unique<pipeline::VisibleRtspPipeline>(external_cfg);

    // 注入全局共享 NPU 推理引擎（多个管线共享同一 InferenceService，节省 NPU 资源）
    if (context.inference_service) {
        context.external_rtsp_pipeline->setInferenceService(context.inference_service.get());
    }

    // 流 ID=1：用于数据库统计溯源和推理回调路由
    context.external_rtsp_pipeline->setStreamId(STREAM_ID_EXTERNAL_RTSP);

    // 推理结果回调：经闭包链 -> 异步分发器落库 + Mosaic 检测事件推送
    context.external_rtsp_pipeline->setInferenceCallback(inference_writer(STREAM_ID_EXTERNAL_RTSP));

    // 帧推送回调：处理后帧 -> FrameHub（供 Mosaic 拼接消费）
    //lambda表达式：回调函数Callback_func([参数捕获列表](回调参数列表){code 回调函数本体})
    context.external_rtsp_pipeline->setFrameCallback(
        [&context](const pipeline::FrameHub::DmaFrame& frame, std::chrono::system_clock::time_point ts,
                   int64_t mono_ns, const pipeline::FrameHub::FrameOverlay& overlay) {
            publishDmaFrameToHub(context, pipeline::FrameSource::ExternalRtsp, frame, ts, mono_ns, overlay);
        });

    if (!pipeline_start(context.external_rtsp_pipeline) &&
        !retry(context.external_rtsp_pipeline, "ExternalRtsp")) {
        context.external_rtsp_pipeline.reset();
    }
}

// =============================================================================
// initImx415Pipeline —— 初始化 IMX415 V4L2 摄像头管线（B路）
// 流程：加载配置 -> 实例化管线 -> 注入推理服务 -> 绑定推理回调/帧推送回调 -> 启动
// IMX415 仅重试 1 次：MIPI 摄像头硬件故障重试意义不大
// =============================================================================
template<typename InferenceWriter>
void AppInitializer::initImx415Pipeline(AppContext& context, const std::string& cfg_path,
                                         InferenceWriter inference_writer) {
    pipeline::V4l2CameraConfig config;
    if (!pipeline::V4l2CameraPipeline::loadFromIni(cfg_path, config) || !config.enable) {
        std::cout << "[IMX415] disabled placeholder; set video_source_imx415.enable=true after selecting the V4L2 node\n";
        return;
    }

    context.imx415_cfg = config;
    context.imx415_pipeline = std::make_unique<pipeline::V4l2CameraPipeline>(config);

    context.imx415_pipeline->setInferenceService(context.inference_service.get());
    context.imx415_pipeline->setInferenceCallback(inference_writer(STREAM_ID_IMX415));
    context.imx415_pipeline->setFrameCallback(
        [&context](const pipeline::FrameHub::DmaFrame& frame, std::chrono::system_clock::time_point ts,
                   int64_t mono_ns, const pipeline::FrameHub::FrameOverlay& overlay) {
            publishDmaFrameToHub(context, pipeline::FrameSource::Imx415, frame, ts, mono_ns, overlay);
        });

    if (!pipeline_start(context.imx415_pipeline) &&
        !retry(context.imx415_pipeline, "IMX415", 1, 1000)) {
        context.imx415_pipeline.reset();
    }
}

// =============================================================================
// initMosaicPipeline —— 初始化多路拼接推流管线
// 从 FrameHub 取帧 -> 拼接 -> 编码 -> RTSP 推流
// 注册事件录制完成回调：录像片段完成后将元数据写入 SQLite event_record 表
// =============================================================================
void AppInitializer::initMosaicPipeline(AppContext& context, const std::string& cfg_path) {
    pipeline::MosaicStreamConfig mosaic_cfg;
    if (!pipeline::MosaicStreamPipeline::loadFromIni(cfg_path, mosaic_cfg) || !mosaic_cfg.enable) return;

    context.mosaic_cfg = mosaic_cfg;
    context.mosaic_pipeline = std::make_unique<pipeline::MosaicStreamPipeline>(mosaic_cfg, context.frame_hub);

    // 事件录制完成回调：将录像文件元信息写入 event_record 表
    context.mosaic_pipeline->setRecordingCompletionCallback(
        [&context](const recording::RecordingMetadata& metadata) {
            if (!context.storage_enabled) return;
            std::ostringstream payload;
            payload << "{\"file_path\":\"" << metadata.file_path
                    << "\",\"event_types\":\"" << metadata.event_types
                    << "\",\"start_real_ms\":" << metadata.start_real_ms
                    << ",\"end_real_ms\":" << metadata.end_real_ms
                    << ",\"trigger_real_ms\":" << metadata.trigger_real_ms
                    << ",\"file_size_bytes\":" << metadata.file_size_bytes
                    << ",\"complete\":" << (metadata.complete ? "true" : "false") << '}';
            context.sqlite_store.insertRawRecord("event_record", metadata.trigger_real_ms, payload.str());
            if (context.mqtt_enabled) context.mqtt_uploader.notifyNewData();
        });

    if (!pipeline_start(context.mosaic_pipeline) &&
        !retry(context.mosaic_pipeline, "Mosaic")) {
        context.mosaic_pipeline.reset();
    }
}

// =============================================================================
// 显式模板实例化：为编译器生成 InferenceWriter 闭包类型的符号
// =============================================================================
template void AppInitializer::initVideoPipelines(
    AppContext&, const std::string&,
    decltype(createInferenceWriter(std::declval<AppContext&>())));
template void AppInitializer::initExternalRtspPipeline(
    AppContext&, const std::string&,
    decltype(createInferenceWriter(std::declval<AppContext&>())));
template void AppInitializer::initImx415Pipeline(
    AppContext&, const std::string&,
    decltype(createInferenceWriter(std::declval<AppContext&>())));
