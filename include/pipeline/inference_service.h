#pragma once  // 头文件保护宏，防止重复包含

// 标准库头文件
#include <array>       // std::array 固定大小数组容器
#include <atomic>      // std::atomic 原子操作类型
#include <chrono>      // std::chrono 时间工具
#include <cstdint>     // int64_t / uint8_t 等定宽整数类型
#include <memory>      // std::unique_ptr 独占所有权智能指针
#include <mutex>       // std::mutex 互斥锁
#include <set>         // std::set 集合容器，用于类别白名单过滤
#include <shared_mutex> // std::shared_mutex 服务生命周期读写锁
#include <string>      // std::string 字符串类型
#include <vector>      // std::vector 动态数组容器

// OpenCV 头文件
#include <opencv2/core/mat.hpp>               // cv::Mat 图像矩阵

// 项目内部头文件
#include "data_processing/postprocess.h"       // detect_result_group_t 检测结果结构体
#include "pipeline/rga_preprocessor.h"         // RgaPixelFormat DMA输入格式
#include "rknn_api.h"                          // Rockchip NPU API (rknn_core_mask 等)

class rknn_lite;                               // 前向声明 RKNN Lite 推理运行时封装类
namespace utils { class NpuThermalManager; }   // 前向声明 NPU 热管理类

namespace pipeline {  // 管线命名空间

/// 最大推理模型数量（Coco + Fire + Ppe + 1 备用槽位）
constexpr size_t kMaxInferenceModels = 4;

/**
 * @enum InferenceModelId
 * @brief 推理模型标识枚举
 *
 * 三个模型槽位：COCO（人员/猫狗）、Fire（火焰/烟雾）、PPE（安全装备）
 * 额外保留 Spare 备用槽位。
 */
enum class InferenceModelId : uint8_t {
    Coco = 0,  // COCO 模型索引：人员/猫狗检测
    Fire = 1,  // Fire 模型索引：火焰/烟雾检测
    Ppe = 2,   // PPE 模型索引：安全装备检测
    Spare = 3  // 备用槽位索引：预留扩展
};

/**
 * @struct InferenceResult
 * @brief 推理结果结构体
 *
 * 封装一次推理（可能包含多个模型）的完整输出。
 */
struct InferenceResult {
    std::array<detect_result_group_t, kMaxInferenceModels> detections{}; // 各模型的检测结果数组
    bool success = false;         // 推理是否成功完成
    uint8_t updated_mask = 0;     // 位掩码标记哪些模型被更新（bit[model_id]=1 表示已更新）
    bool skipped = false;         // 是否被跳帧跳过（未实际推理）
    std::string skip_reason;      // 跳帧原因描述（用于调试）
};

/**
 * @struct InferenceFrameDesc
 * @brief DMA-BUF 帧描述符
 *
 * 用于零 CPU 像素拷贝推理路径：源 FD 由 RGA 读取，结果直接写入
 * RKNN 绑定的输入 DMA-BUF。
 */
struct InferenceFrameDesc {
    int dma_fd = -1;           // DMA-BUF 文件描述符（-1 表示无效/未设置）
    int width = 0;             // 帧宽度（像素）
    int height = 0;            // 帧高度（像素）
    int width_stride = 0;      // 行步长（字节），对齐后的每行尺寸
    int height_stride = 0;     // 高度步长（像素），对齐后的垂直尺寸
    size_t buffer_size = 0;    // 缓冲区总大小（字节）
    RgaPixelFormat pixel_format = RgaPixelFormat::NV12; // NV12/NV16 DMA格式
    int scheduler_slot = 0;    // 调度槽位：0 = A路 (external_rtsp), 1 = B路 (imx415)
    std::shared_ptr<void> frame_hold; // 保持源 DMA-BUF 到所有模型 RGA 读取完成
    /// 检查描述符是否有效（DMA fd 有效且尺寸非零）
    bool valid() const { return dma_fd >= 0 && width > 0 && height > 0; }
};

/**
 * @struct InferenceModelConfig
 * @brief 单个推理模型的配置
 *
 * 描述模型路径、标签文件、类别过滤等参数。
 */
struct InferenceModelConfig {
    std::string name;                // 模型名称（用于日志标识）
    std::string path;                // RKNN 模型文件路径
    std::string label_path;          // 标签文件路径（每行一个类别名）
    int class_count = 1;             // 该模型的类别总数
    // 按源流的类别白名单：[0]=Stream A (external_rtsp), [1]=Stream B (imx415)
    // 空集 = 允许所有类别（不过滤）
    std::array<std::set<int>, 2> stream_class_filter; // 两路流各自的类别 ID 白名单
    float confidence_threshold = 0.40F;  // 置信度阈值（低于此值的检测结果被丢弃）
    float nms_threshold = 0.50F;         // NMS（非极大值抑制）的 IoU 阈值
};

/**
 * @struct InferenceServiceConfig
 * @brief 推理服务的总体配置
 *
 * 包含所有模型的配置、每路流使用哪些模型的位掩码等。
 */
struct InferenceServiceConfig {
    std::array<InferenceModelConfig, kMaxInferenceModels> models{}; // 各模型的配置数组
    // slot 0=external RTSP (A路), slot 1=IMX415 (B路). Each bit selects a model.
    // 默认: A路=Coco+Ppe (0x05), B路=Coco+Fire (0x03)
    std::array<uint8_t, 2> source_model_masks{{0x05, 0x03}};       // 每路流启用哪些模型的位掩码
    std::array<int, 2> source_models_per_frame{{2, 2}};             // 每路流每帧运行的模型数量
};

/**
 * @class InferenceService
 * @brief 全局推理服务
 *
 * 管理所有 RKNN 模型的加载、推理调度和结果聚合。
 *
 * 职责：
 *   1. 加载多个 .rknn 模型到 NPU 内存
 *   2. 提供 infer() / inferFromFd() 统一的推理接口
 *   3. 轮询调度多路源流的多模型推理
 *   4. 收集各模型的运行时统计
 *   5. 结合 NPU 热管理进行频率/电压控制
 *
 * 线程安全：内部使用 NPU 全局互斥锁和每模型独立锁。
 */
class InferenceService {
public:
    /**
     * @struct ModelRuntimeStats
     * @brief 模型运行时统计
     *
     * 公开的只读统计快照，供外部监控使用。
     */
    struct ModelRuntimeStats {
        std::array<uint64_t, kMaxInferenceModels> count{};      // 各模型累计推理次数
        std::array<int64_t, kMaxInferenceModels> last_us{};     // 各模型最近一次推理耗时（微秒）
        std::array<int64_t, kMaxInferenceModels> average_us{};  // 各模型平均推理耗时（微秒）
        std::array<int64_t, kMaxInferenceModels> max_us{};      // 各模型最大推理耗时（微秒）
    };

    InferenceService();   // 默认构造函数
    ~InferenceService();  // 析构函数（自动调用 shutdown）

    // 禁止拷贝（管理硬件资源，不可复制）
    InferenceService(const InferenceService&) = delete;
    InferenceService& operator=(const InferenceService&) = delete;

    /**
     * @brief 初始化推理服务
     * @param config 推理服务配置
     * @return true 所有模型加载成功；false 任一个模型加载失败
     *
     * 加载配置中的所有 .rknn 模型，初始化 NPU 推理上下文。
     */
    bool initialize(const InferenceServiceConfig& config);

    /// 关闭推理服务，释放所有模型资源和 NPU 上下文
    void shutdown();

    /**
     * @brief 执行推理（从 cv::Mat 输入）
     * @param frame          输入图像（BGR 格式）
     * @param scheduler_slot 调度槽位：0=A路, 1=B路
     * @return 推理结果（包含所有启用的模型检测结果）
     *
     * 根据配置的 source_model_masks[scheduler_slot] 选择启用的模型，
     * 轮询执行推理并将结果聚合到 InferenceResult 中。
     */
    InferenceResult infer(const cv::Mat& frame, int scheduler_slot = 0);

    /**
     * @brief 执行推理（从 DMA-BUF fd 输入，零拷贝路径）
     * @param desc DMA-BUF 帧描述符
     * @return 推理结果
     *
     * 通过 RGA 把源 DMA-BUF 转换到 RKNN 输入 DMA-BUF，避免中间
     * cv::Mat 和 rknn_inputs_set 输入复制。
     */
    InferenceResult inferFromFd(InferenceFrameDesc desc);

    /// 推理服务是否已初始化并可用
    bool isReady() const { return initialized_.load(std::memory_order_acquire); }

    /// 获取运行统计的原子快照
    ModelRuntimeStats statsSnapshot() const;

    /// 设置 NPU 核心掩码（控制使用哪些 NPU 核心）
    void setCoreMask(rknn_core_mask mask);

    /// 注入 NPU 热管理器（用于根据温度调整 NPU 频率）
    void setThermalManager(utils::NpuThermalManager* mgr) { thermal_mgr_ = mgr; }

private:
    /**
     * @brief 推理内部实现
     * @param frame          cv::Mat 输入（可为 nullptr 表示使用 fd 输入）
     * @param desc           DMA-BUF 描述符（可为 nullptr 表示使用 Mat 输入）
     * @param scheduler_slot 调度槽位
     * @return 推理结果
     *
     * 完成 RGA 预处理、模型推理、后处理的全流程。
     */
    InferenceResult inferImpl(const cv::Mat* frame, InferenceFrameDesc* desc, int scheduler_slot);

    InferenceServiceConfig config_;                                    // 推理服务配置副本
    std::atomic<bool> initialized_{false};                              // 初始化完成标志
    std::array<std::unique_ptr<rknn_lite>, kMaxInferenceModels> models_; // 各模型的运行时实例（unique_ptr 独占所有权）
    mutable std::shared_mutex lifecycle_mutex_;                          // 防止推理期间释放模型
    std::mutex npu_mutex_;                                             // NPU 全局互斥锁（确保推理串行执行）
    std::array<std::mutex, kMaxInferenceModels> model_mutex_;          // 每个模型的独立锁（保护模型特定状态）
    utils::NpuThermalManager* thermal_mgr_ = nullptr;                  // NPU 热管理器指针（非持有）
    std::array<std::atomic<uint64_t>, 2> source_round_robin_{};        // 每路流模型的轮询计数器
    // 以下为模型运行时统计（原子变量，线程安全）
    std::array<std::atomic<uint64_t>, kMaxInferenceModels> model_count_{};      // 各模型推理次数
    std::array<std::atomic<uint64_t>, kMaxInferenceModels> model_total_us_{};  // 各模型累计推理耗时（微秒）
    std::array<std::atomic<int64_t>, kMaxInferenceModels> model_last_us_{};   // 各模型最近一次推理耗时（微秒）
    std::array<std::atomic<int64_t>, kMaxInferenceModels> model_max_us_{};    // 各模型最大推理耗时（微秒）
};

}  // namespace pipeline
