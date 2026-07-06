#pragma once

// =============================================================================
// app_initializer.h —— 应用初始化器
//
// AppContext：应用运行期上下文，聚合所有模块实例和配置的"聚合根"结构体。
//   避免全局变量，便于测试和多实例场景。
//
// AppInitializer：静态工具类，按依赖关系初始化/关闭所有子系统。
//
// 初始化顺序（严格不可乱）:
//   线程RT配置 -> 温控 -> 配置摘要 -> 存储 -> MQTT -> 分发器 -> 推理服务 -> FrameHub -> 视频管线 -> 拓扑校验 -> 温控启动
// 关闭顺序（逆序）:
//   视频管线 -> 推理服务 -> 分发器 -> 温控 -> MQTT -> 存储
// =============================================================================

#include <cstdint>
#include <memory>
#include <string>

#include "app/inference_result_dispatcher.h"
#include "network/data_upload/mqtt_uploader.h"
#include "monitoring/board_status_collector.h"
#include "pipeline/frame_hub.h"
#include "pipeline/inference_service.h"
#include "pipeline/mosaic_stream_pipeline.h"
#include "pipeline/v4l2_camera_pipeline.h"
#include "pipeline/visible_rtsp_pipeline.h"
#include "storage/sqlite_store.h"
#include "utils/npu_thermal_manager.h"

// =============================================================================
// AppContext —— 应用运行期上下文（聚合根）
// =============================================================================
struct AppContext {
    // ---- 存储 ----
    data_lifecycle::SqliteStore sqlite_store;
    data_lifecycle::StorageConfig storage_cfg;
    bool storage_enabled = false;

    // ---- MQTT 上报 ----
    network::MqttUploader mqtt_uploader;
    network::MqttConfig mqtt_cfg;
    bool mqtt_enabled = false;

    // ---- 推理结果异步分发 ----
    std::unique_ptr<app::InferenceResultDispatcher> inference_dispatcher;
    int inference_sample_ms = 0;  // 推理采样周期（毫秒）；0=全量落库

    // ---- 板端监控 ----
    monitoring::BoardStatusCollector board_status_collector;

    // ---- 帧汇聚与推理 ----
    std::shared_ptr<pipeline::FrameHub> frame_hub;                // 多路帧中转站
    std::shared_ptr<pipeline::InferenceService> inference_service; // 全局共享 NPU 推理引擎

    // ---- 视频管线 ----
    pipeline::VisibleRtspPipelineConfig external_rtsp_cfg;
    std::unique_ptr<pipeline::VisibleRtspPipeline> external_rtsp_pipeline;  // A路：RTSP拉流+推理

    pipeline::V4l2CameraConfig imx415_cfg;
    std::unique_ptr<pipeline::V4l2CameraPipeline> imx415_pipeline;          // B路：IMX415采集+推理

    pipeline::MosaicStreamConfig mosaic_cfg;
    std::unique_ptr<pipeline::MosaicStreamPipeline> mosaic_pipeline;        // 拼接推流

    // ---- NPU 温控 ----
    // 通过 sysfs 读取 thermal_zone 温度，分级触发降频策略，注入 InferenceService 形成热保护闭环
    utils::NpuThermalManager thermal_manager;
};

// =============================================================================
// AppInitializer —— 应用初始化与收尾管理
// =============================================================================
class AppInitializer {
public:
    // 串联初始化所有子系统（按依赖顺序）；返回 false 表示关键组件失败
    static bool initialize(AppContext& context);

    // 按逆序释放所有资源
    static void shutdown(AppContext& context);

private:
    // ---- 基础服务（无前置依赖） ----
    static void initStorage(AppContext& context, const std::string& cfg_path);
    static void initMqtt(AppContext& context, const std::string& cfg_path);  // 依赖存储可用

    // ---- 推理服务 ----
    static void initInferenceService(AppContext& context, const std::string& cfg_path);

    // ---- 视频管线（模板方法：支持注入推理写入器闭包） ----
    template<typename InferenceWriter>
    static void initVideoPipelines(AppContext& context, const std::string& cfg_path,
                                    InferenceWriter inference_writer);

    template<typename InferenceWriter>
    static void initExternalRtspPipeline(AppContext& context, const std::string& cfg_path,
                                          InferenceWriter inference_writer);

    template<typename InferenceWriter>
    static void initImx415Pipeline(AppContext& context, const std::string& cfg_path,
                                    InferenceWriter inference_writer);

    static void initMosaicPipeline(AppContext& context, const std::string& cfg_path);

    // ---- 运维辅助 ----
    // N12优化：启动时格式化输出生效配置，便于快速排查问题
    static void printConfigSummary(const std::string& cfg_path);
};
