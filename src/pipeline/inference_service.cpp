/**
 * @file inference_service.cpp
 * @brief AI 推理服务实现
 *
 * 该文件实现了 InferenceService 类，是整个系统的核心推理引擎。
 *
 * 主要职责：
 * 1. 管理多个 RKNN 模型（最多 kMaxInferenceModels 个）的生命周期
 * 2. 提供统一的推理接口：infer()（cv::Mat 路径）和 inferFromFd()（DMA-BUF FD 路径）
 * 3. 实现模型轮转调度（Round-Robin）：多模型按源类型路由，每帧轮转执行部分模型
 * 4. 支持同类过滤（stream_class_filter）：按流类型过滤检测结果的类别
 * 5. 支持共享预处理缓冲区：同帧首个模型的 RGA 预处理结果可复用，减少重复计算
 * 6. 提供推理性能统计：次数、平均耗时、最大耗时、最近耗时
 * 7. 支持 NPU 热管理（thermal_mgr_）：过热时自动跳帧
 *
 * 模型路由机制（source_model_masks）：
 * - 每个源（source 0: ExternalRtsp, source 1: Imx415）有一个模型掩码
 * - 掩码的每一位对应一个模型槽位（bit 0 → models_[0]）
 * - source_models_per_frame 控制每帧执行该源中的几个模型
 * - Round-Robin 轮转确保所有模型都有机会执行
 *
 * 零拷贝推理路径（inferFromFd）：
 * - 从 MPP 硬件解码器获取 DMA-BUF FD
 * - RGA 从解码 FD 读取 NV12，直接写入 RKNN 绑定的 RGB DMA-BUF
 * - NPU 从绑定输入 DMA-BUF 读取，不经过中间 cv::Mat/rknn_inputs_set
 * - RGA 预处理由 RKNN 模型内部完成
 */
#include "pipeline/inference_service.h"

#include <algorithm>  // std::min, std::max, std::clamp
#include <iostream>   // 控制台输出
#include <vector>     // std::vector

#include "utils/npu_thermal_manager.h" // NPU 热管理（过热保护）
#include "utils/rknnPool.hpp"          // RKNN 模型池封装

namespace pipeline {

// ============================================================================
// 构造/析构
// ============================================================================
InferenceService::InferenceService() = default;      // 默认构造
InferenceService::~InferenceService() { shutdown(); } // 析构时自动关闭

// ============================================================================
// initialize: 初始化推理服务，加载所有已配置的 RKNN 模型
// @param config 推理服务配置（模型列表、路由掩码、每帧模型数等）
// @return       true 表示至少加载成功一个模型
//
// 加载流程：
// 1. 遍历 models_ 数组（最多 kMaxInferenceModels 个槽位）
// 2. 跳过路径为空的模型配置
// 3. 使用 rknn_lite 构造器加载每个模型
// 4. 设置模型参数：核心数、类别数、标签路径、置信度/NMS 阈值
// 5. 验证模型有效性
// 6. 输出路由配置
// ============================================================================
bool InferenceService::initialize(const InferenceServiceConfig& config) {
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (initialized_.load(std::memory_order_acquire)) return true; // 已初始化，幂等返回
    config_ = config;
    bool any_loaded = false; // 是否至少加载了一个模型

    // 遍历模型槽位并加载
    for (size_t i = 0; i < models_.size(); ++i) {
        const auto& spec = config_.models[i]; // 第 i 个模型的配置
        if (spec.path.empty()) continue;      // 空路径 → 跳过

        try {
            // 创建 rknn_lite 实例（封装 RKNN 模型加载和推理）
            // 参数：模型路径, 核心数(0=自动), 类别数, 模型索引, NPU核心策略, 标签路径, 置信度阈值, NMS阈值
            models_[i] = std::make_unique<rknn_lite>(
                const_cast<char*>(spec.path.c_str()),           // 模型文件路径
                0,                                               // 核心数（0 表示自动）
                std::max(1, spec.class_count),                   // 类别数（至少 1）
                static_cast<int>(i),                             // 模型槽位索引
                RKNN_NPU_CORE_AUTO,                              // NPU 核心自动分配
                spec.label_path,                                 // 标签文件路径
                spec.confidence_threshold,                       // 置信度阈值
                spec.nms_threshold);                             // NMS 阈值

            // 验证模型是否有效
            if (!models_[i]->isValid()) {
                std::cerr << "[InferenceService] invalid model " << spec.name
                          << ": " << spec.path << '\n';
                models_[i].reset(); // 释放无效模型
                continue;
            }

            any_loaded = true; // 标记至少加载了一个模型
            std::cout << "[InferenceService] loaded model=" << spec.name
                      << " labels=" << spec.label_path
                      << " classes=" << spec.class_count
                      << " slot=" << i << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[InferenceService] load failed " << spec.name
                      << ": " << error.what() << '\n';
            models_[i].reset(); // 异常时释放
        }
    }

    // 输出路由配置
    for (size_t source = 0; source < config_.source_model_masks.size(); ++source) {
        std::cout << "[InferenceService] route source=" << source
                  << " mask=0x" << std::hex << static_cast<int>(config_.source_model_masks[source])
                  << std::dec << " models_per_frame=" << config_.source_models_per_frame[source] << '\n';
    }

    initialized_.store(any_loaded, std::memory_order_release); // 至少一个模型可用才算初始化成功
    return initialized_.load(std::memory_order_acquire);
}

// ============================================================================
// shutdown: 关闭推理服务，释放所有模型资源
//
// 加锁保护 model 数组的释放操作，防止与正在执行的推理产生竞态。
// ============================================================================
void InferenceService::shutdown() {
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    initialized_.store(false, std::memory_order_release); // 首先阻止新的推理提交
    std::array<std::unique_lock<std::mutex>, kMaxInferenceModels> model_locks{
        std::unique_lock<std::mutex>(model_mutex_[0]),
        std::unique_lock<std::mutex>(model_mutex_[1]),
        std::unique_lock<std::mutex>(model_mutex_[2]),
        std::unique_lock<std::mutex>(model_mutex_[3])};
    std::lock_guard<std::mutex> lock(npu_mutex_); // 锁顺序固定为 model -> NPU
    for (auto& model : models_) model.reset();    // 逐一释放所有模型
}

// ============================================================================
// infer: 执行推理（cv::Mat 输入路径）
// @param frame          输入 BGR888 图像
// @param scheduler_slot 调度槽位（0=ExternalRtsp, 1=Imx415），决定使用哪些模型的路由掩码
// @return               推理结果（包含检测框、更新掩码等）
//
// 委托给 inferImpl() 统一执行。
// ============================================================================
InferenceResult InferenceService::infer(const cv::Mat& frame, int scheduler_slot) {
    if (frame.empty()) {
        // 空帧处理：返回跳过状态
        InferenceResult result;
        result.skipped = true;
        result.skip_reason = "empty_frame";
        return result;
    }
    return inferImpl(&frame, nullptr, scheduler_slot); // 委托实现
}

// ============================================================================
// inferFromFd: 执行推理（DMA-BUF FD 输入路径，零拷贝）
// @param desc 帧描述符（包含 DMA-BUF FD、尺寸、stride 等）
// @return     推理结果
//
// 委托给 inferImpl() 统一执行。RGA 将 NV12 FD 直接转换到 RKNN 输入 DMA-BUF，
// NPU 从绑定缓冲读取 RGB tensor，不经过中间 cv::Mat。
// ============================================================================
InferenceResult InferenceService::inferFromFd(InferenceFrameDesc desc) {
    if (!desc.valid()) {
        // 无效描述符：返回跳过状态
        InferenceResult result;
        result.skipped = true;
        result.skip_reason = "invalid_frame_desc";
        return result;
    }
    return inferImpl(nullptr, &desc, desc.scheduler_slot); // 委托实现
}

// ============================================================================
// inferImpl: 推理核心实现（统一处理 cv::Mat 和 FD 两种输入路径）
// @param frame          cv::Mat 帧指针（cv::Mat 路径使用，FD 路径为 nullptr）
// @param desc           帧描述符指针（FD 路径使用，cv::Mat 路径为 nullptr）
// @param scheduler_slot 调度槽位（0 或 1）
// @return               推理结果
//
// 完整推理流程：
// 1. 前置检查：服务就绪、热管理
// 2. 路由计算：根据 source_model_masks 确定需要运行的模型列表
// 3. Round-Robin 选择：从活跃模型中轮转选取 run_count 个
// 4. 逐个模型推理（加锁保护每个模型的 RKNN 上下文）
// 5. 共享预处理缓冲区：同帧首模型的预处理结果后续模型可复用
// 6. 类别过滤：按 source 的 stream_class_filter 过滤结果
// 7. 性能统计：记录耗时、次数、最大值等
// ============================================================================
InferenceResult InferenceService::inferImpl(const cv::Mat* frame,
                                            InferenceFrameDesc* desc,
                                            int scheduler_slot) {
    InferenceResult result;
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);

    // ---- 1. 前置检查 ----
    if (!initialized_.load(std::memory_order_acquire)) {
        result.skipped = true;
        result.skip_reason = "service_not_ready"; // 服务未就绪
        return result;
    }

    // 热管理：NPU 过热时跳过推理
    if (thermal_mgr_ && thermal_mgr_->shouldSkipInference()) {
        result.skipped = true;
        result.skip_reason = "thermal_throttle"; // 热节流
        return result;
    }

    // ---- 2. 路由计算 ----
    // 将 scheduler_slot 钳制在 0-1 范围，确定是哪个源
    const size_t source = static_cast<size_t>(std::clamp(scheduler_slot, 0, 1));
    // 获取该源对应的模型路由掩码（bit i = 1 表示 models_[i] 参与处理此源）
    const uint8_t route_mask = config_.source_model_masks[source];

    // 收集路由掩码中启用的模型的索引列表
    std::vector<size_t> active;
    for (size_t i = 0; i < models_.size(); ++i) {
        if (models_[i] && (route_mask & (1u << i)) &&
            (!desc || models_[i]->supportsFdInput())) {
            active.push_back(i);
        }
    }
    if (active.empty()) {
        result.skipped = true;
        result.skip_reason = desc ? "no_fd_model_for_source" : "no_model_for_source";
        return result;
    }

    // ---- 3. Round-Robin 轮转选择 ----
    // 每帧运行 run_count 个模型（不超过活跃模型总数）
    const size_t run_count = std::min(active.size(), static_cast<size_t>(
        std::max(1, config_.source_models_per_frame[source])));
    // 原子递增轮转计数并计算起始索引
    const size_t start = source_round_robin_[source].fetch_add(run_count) % active.size();

    std::vector<size_t> selected;
    selected.reserve(run_count);
    for (size_t offset = 0; offset < run_count; ++offset) {
        selected.push_back(active[(start + offset) % active.size()]);
    }

    auto finish_model = [&](size_t index, bool ok, int64_t elapsed) {
        model_count_[index].fetch_add(1);                            // 累计推理次数
        model_total_us_[index].fetch_add(static_cast<uint64_t>(elapsed)); // 累计总耗时
        model_last_us_[index].store(elapsed);                        // 最近一次耗时
        // 更新最大耗时（使用 CAS 循环保证原子性）
        auto current_max = model_max_us_[index].load();
        while (elapsed > current_max &&
               !model_max_us_[index].compare_exchange_weak(current_max, elapsed)) {}

        if (!ok) return;

        // ---- 5. 类别过滤（按流类型过滤） ----
        // 根据 source 的 stream_class_filter 白名单过滤检测结果
        const auto& class_filter = config_.models[index].stream_class_filter[source];
        if (!class_filter.empty()) {
            // 白名单非空 → 过滤
            auto& dets = result.detections[index]; // 引用该模型的检测结果
            int kept = 0; // 保留的检测结果计数
            for (int item = 0; item < dets.count; ++item) {
                // 检查该类 ID 是否在白名单中
                if (class_filter.count(dets.results[item].class_id)) {
                    if (kept != item) dets.results[kept] = dets.results[item]; // 移动保留项到前面
                    ++kept;
                }
            }
            dets.count = kept; // 更新有效检测结果数量
        }

        // 标记该模型有更新
        result.updated_mask |= static_cast<uint8_t>(1u << index); // 设置位掩码
        result.success = true; // 至少一个模型成功
    };

    if (desc) {
        // Acquire all per-model locks in stable index order. Holding them across the
        // prepare/run split prevents another source from overwriting a prepared input.
        std::vector<size_t> lock_order = selected;
        std::sort(lock_order.begin(), lock_order.end());
        std::vector<std::unique_lock<std::mutex>> model_locks;
        model_locks.reserve(lock_order.size());
        for (size_t index : lock_order) model_locks.emplace_back(model_mutex_[index]);

        std::array<bool, kMaxInferenceModels> prepared{};
        std::array<int64_t, kMaxInferenceModels> elapsed_us{};
        for (size_t index : selected) {
            const auto started = std::chrono::steady_clock::now();
            prepared[index] = models_[index]->prepareFromFd(
                desc->dma_fd, desc->width, desc->height,
                desc->width_stride, desc->height_stride, desc->buffer_size,
                desc->pixel_format) == 0;
            elapsed_us[index] += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - started).count();
        }

        // Every synchronous RGA submission has returned. No selected model accesses
        // the decoder buffer beyond this point, so MPP may recycle it while NPU runs.
        desc->frame_hold.reset();

        for (size_t index : selected) {
            bool ok = false;
            if (prepared[index]) {
                const auto started = std::chrono::steady_clock::now();
                ok = models_[index]->inferPrepared(
                    result.detections[index], &npu_mutex_) == 0;
                elapsed_us[index] += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - started).count();
            }
            finish_model(index, ok, elapsed_us[index]);
            const auto lock_it = std::lower_bound(lock_order.begin(), lock_order.end(), index);
            if (lock_it != lock_order.end() && *lock_it == index) {
                model_locks[static_cast<size_t>(lock_it - lock_order.begin())].unlock();
            }
        }
        return result;
    }

    // cv::Mat path retains shared preprocessing and locks one model at a time.
    cv::Mat shared_preproc;
    for (size_t index : selected) {
        std::lock_guard<std::mutex> model_lock(model_mutex_[index]);
        const auto started = std::chrono::steady_clock::now();
        if (!shared_preproc.empty()) models_[index]->setSharedPreproc(shared_preproc);
        models_[index]->ori_img = *frame;
        const bool ok = models_[index]->interf(
            result.detections[index], false, &npu_mutex_) == 0;
        if (shared_preproc.empty() && ok) {
            shared_preproc = models_[index]->getPreprocessedBuf();
        }
        const int64_t elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
        finish_model(index, ok, elapsed);
    }
    return result;
}

// ============================================================================
// setCoreMask: 设置所有模型的 NPU 核心掩码
// @param mask RKNN 核心掩码（控制使用哪些 NPU 核心）
//
// 运行时动态调整 NPU 核心分配策略。
// ============================================================================
void InferenceService::setCoreMask(rknn_core_mask mask) {
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    std::array<std::unique_lock<std::mutex>, kMaxInferenceModels> model_locks{
        std::unique_lock<std::mutex>(model_mutex_[0]),
        std::unique_lock<std::mutex>(model_mutex_[1]),
        std::unique_lock<std::mutex>(model_mutex_[2]),
        std::unique_lock<std::mutex>(model_mutex_[3])};
    std::lock_guard<std::mutex> lock(npu_mutex_); // 与推理保持一致的锁顺序，避免死锁
    for (auto& model : models_) if (model) model->setCoreMask(mask); // 逐一设置
}

// ============================================================================
// statsSnapshot: 获取推理性能统计快照（线程安全）
// @return ModelRuntimeStats 结构体，包含每模型的：
//         - count: 累计推理次数
//         - last_us: 最近一次推理耗时（微秒）
//         - max_us: 最大推理耗时（微秒）
//         - average_us: 平均推理耗时（微秒）
// ============================================================================
InferenceService::ModelRuntimeStats InferenceService::statsSnapshot() const {
    ModelRuntimeStats stats;
    for (size_t i = 0; i < stats.count.size(); ++i) {
        stats.count[i] = model_count_[i].load();       // 累计次数
        stats.last_us[i] = model_last_us_[i].load();   // 最近耗时
        stats.max_us[i] = model_max_us_[i].load();     // 最大耗时
        const uint64_t total = model_total_us_[i].load(); // 累计总耗时
        stats.average_us[i] = stats.count[i] == 0 ? 0 :
            static_cast<int64_t>(total / stats.count[i]); // 平均耗时 = 总耗时 / 次数
    }
    return stats;
}

}  // namespace pipeline
