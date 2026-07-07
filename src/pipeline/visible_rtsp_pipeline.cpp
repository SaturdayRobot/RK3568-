/**
 * @file visible_rtsp_pipeline.cpp
 * @brief RTSP 可见光处理管线实现
 *
 * 这条管线对应"网络视频输入"路径，整体职责可分成两段：
 *
 * 1. StreamLoader 线程（loader_thread_）：
 *    - 用 FFmpeg 从 RTSP 源拉取码流包
 *    - 把 H264/H265 码流送入 MPP 硬件解码器
 *    - 以 DMA-BUF FD（文件描述符）的形式把硬件解码帧交给上层消费者
 *    - 支持断线自动重连（指数退避策略）
 *
 * 2. processingLoop 线程（processing_thread_）：
 *    - 决定是否触发 NPU 推理（按 inference_interval_ms 间隔）
 *    - 必要时通过 RGA 把 NV12 硬件帧转成 BGR 格式（用于 OpenCV 处理和显示）
 *    - 执行检测结果融合、帧回调分发、推理统计
 *    - 将处理后的帧通过 frame_callback_ 传给 FrameHub
 *
 * 3. inferenceLoop 线程（inference_thread_）：
 *    - 异步执行 NPU 推理，避免阻塞主处理循环
 *    - 通过条件变量与 processingLoop 通信
 *    - 将推理结果缓存到 detection_cache_ 供处理循环使用
 *
 * 因为 RTSP 拉流抖动和业务处理耗时特征不同，所以和本地 V4L2 管线分成了独立实现，
 * 但对外提供的"回调 / 推流 / 归档 / 推理统计"接口保持一致，方便上层统一编排。
 */
#include "pipeline/visible_rtsp_pipeline.h"

#include <algorithm>    // std::max, std::min
#include <chrono>       // 高精度时间测量
#include <cstring>      // 字符串操作
#include <iostream>     // 控制台输出
#include <utility>      // std::move

#include <opencv2/imgproc.hpp>           // OpenCV 图像处理

#include "pipeline/inference_service.h"  // AI 推理服务
#include "pipeline/rga_preprocessor.h"   // RGA 硬件预处理
#include "pipeline/detection_stabilizer.h" // 检测结果稳定器（去抖/滤波）
#include "utils/thread_runtime.h"         // 线程运行时设置

using pipeline::ScopedTimer; // RAII 性能计时器

namespace {

// ============================================================================
// DecodedFrameGuard: RAII 守卫类，确保解码帧在使用后被正确释放
// 析构时自动调用 StreamLoader::releaseDecodedFrame() 归还帧引用
// ============================================================================
struct DecodedFrameGuard {
    StreamLoader* loader = nullptr;  // 关联的流加载器
    uint64_t desc_id = 0;           // 帧描述符 ID（0 表示无效）
    ~DecodedFrameGuard() {
        if (loader && desc_id != 0) {
            loader->releaseDecodedFrame(desc_id); // 归还帧引用给 MPP 缓冲池
        }
    }
};

} // unnamed namespace

namespace pipeline {

/**
 * @brief 构造函数，初始化可见光 RTSP 处理管道
 * @param config 可见光 RTSP 管道配置参数（URL、推理参数、拉流参数等）
 *
 * 初始化配置并将内部指标名称设为 "rtsp" 便于日志区分。
 */
VisibleRtspPipeline::VisibleRtspPipeline(VisibleRtspPipelineConfig config)
    : config_(std::move(config)) {
    metrics_.name = "rtsp"; // 指标名，用于日志输出区分不同管线
}

// setInferenceService: 注入共享推理服务实例（外部创建，多管线复用）
void VisibleRtspPipeline::setInferenceService(InferenceService* svc) {
    inference_service_ = svc;
}

// decodedQueueDepth: 返回解码帧就绪队列的当前深度（待消费帧数）
std::uint32_t VisibleRtspPipeline::decodedQueueDepth() const {
    if (!loader_) {
        return 0; // 加载器未初始化
    }
    return static_cast<std::uint32_t>(loader_->readyDepth());
}

// decodedQueueLimit: 返回解码帧就绪队列的最大容量
std::uint32_t VisibleRtspPipeline::decodedQueueLimit() const {
    if (!loader_) {
        return static_cast<std::uint32_t>(config_.loader_max_ready_depth); // 配置值
    }
    return static_cast<std::uint32_t>(loader_->maxReadyDepth()); // 实际值
}

// decodedDropTotal: 返回累计丢弃的解码帧总数
std::uint64_t VisibleRtspPipeline::decodedDropTotal() const {
    if (!loader_) {
        return 0;
    }
    return loader_->droppedTotalCount();
}

/**
 * @brief 析构函数，安全停止可见光 RTSP 处理管道
 *
 * 调用 stop() 确保所有线程退出后再释放资源。
 */
VisibleRtspPipeline::~VisibleRtspPipeline() {
    stop();
}

/**
 * @brief 设置帧回调函数
 * @param cb 帧回调函数，传递NV12 DMA描述符、时间戳和叠加元数据
 *
 * 处理后的每一帧都会通过此回调传给 FrameHub 进行后续分发。
 */
void VisibleRtspPipeline::setFrameCallback(DecodedFrameCallback cb) {
    frame_callback_ = std::move(cb);
}

/**
 * @brief 设置推理回调函数
 * @param cb 推理回调函数，签名为 void(const InferenceStats&)
 *
 * 每次 NPU 推理完成后会通过此回调报告推理统计信息（人数、PPE 数等）。
 */
void VisibleRtspPipeline::setInferenceCallback(InferenceCallback cb) {
    inference_callback_ = std::move(cb);
}

/**
 * @brief 设置流 ID
 * @param id 流 ID，用于标识不同的数据流（在多管线场景下区分来源）
 */
void VisibleRtspPipeline::setStreamId(int id) {
    stream_id_ = id;
}

/**
 * @brief 从 INI 配置文件加载可见光 RTSP 管道配置
 * @param path    配置文件路径
 * @param out     输出的配置结构体（引用）
 * @param section 配置节名（默认 "visible_rtsp"）
 * @return        是否加载成功
 *
 * 加载的配置项包括：
 * - 基本参数：enable, url, pull_interval_ms
 * - 拉流器参数：loader_max_ready_depth, loader_rtbufsize_bytes, loader_stimeout_us 等
 * - 推理参数：inference_enable, inference_interval_ms（支持 fallback 到 [visible_inference] 节）
 */
bool VisibleRtspPipeline::loadFromIni(const std::string& path, VisibleRtspPipelineConfig& out, const std::string& section) {
    IniConfig cfg;
    if (!cfg.load(path)) {
        return false; // 配置文件加载失败
    }

    // ---- 基本拉流配置 ----
    out.enable = cfg.getBool(section, "enable", false);                      // 是否启用此管线
    out.url = cfg.getString(section, "url", "");                             // RTSP 源地址
    out.pull_interval_ms = cfg.getInt(section, "pull_interval_ms", 10);      // 拉流间隔（毫秒）
    out.loader_max_ready_depth = cfg.getInt(section, "loader_max_ready_depth", 4); // 解码帧队列最大深度
    out.loader_rtbufsize_bytes = cfg.getInt(section, "loader_rtbufsize_bytes", 1048576); // 接收缓冲区大小（字节）
    out.loader_stimeout_us = cfg.getInt(section, "loader_stimeout_us", 2000000); // 流超时时间（微秒）
    out.loader_max_delay_us = cfg.getInt(section, "loader_max_delay_us", 100000);  // 最大延迟（微秒）
    out.loader_rtsp_transport = cfg.getString(section, "loader_rtsp_transport", "tcp"); // RTSP 传输协议（tcp/udp）

    // ---- 推理配置（支持 fallback 到 [visible_inference] 节） ----
    out.inference.enable = cfg.getBool(section, "inference_enable",
                                      cfg.getBool("visible_inference", "enable", true)); // 是否启用推理
    out.inference.interval_ms = cfg.getInt(section, "inference_interval_ms",
                                          cfg.getInt("visible_inference", "interval_ms", 0)); // 推理间隔

    return true;
}

/**
 * @brief 启动可见光 RTSP 处理管线
 * @return 是否启动成功
 *
 * 按照 1.状态检查 -> 2.参数校验 -> 3.资源初始化 -> 4.硬件拉流开启 -> 5.线程激活 的顺序执行。
 */
bool VisibleRtspPipeline::start() {
    // 步骤 1: 幂等性检查
    // 如果管线已经在运行，直接返回成功，避免重复启动
    if (running_) {
        return true;
    }

    // 清除检测缓存（上次运行的可能已过期）
    detection_cache_valid_ = false;

    // 步骤 2: 关键配置校验
    // 必须有输入源地址才能启动
    if (config_.url.empty()) {
        std::cerr << "[VisRtspPipeline] Start failed: Source URL is empty" << std::endl;
        return false;
    }

    // 步骤 3: 检查共享推理服务
    if (config_.inference.enable) {
        // 确保全局 NPU 推理引擎已注入并就绪
        if (!inference_service_ || !inference_service_->isReady()) {
            std::cerr << "[VisRtspPipeline] Start failed: Inference service not injected or not ready" << std::endl;
            return false;
        }
    }

    // 步骤 4: 硬件拉流引擎初始化 (StreamLoader 内部包含 FFmpeg 拉流与 MPP 硬件解码)
    // 实例化 StreamLoader 对象，传入流地址和流 ID
    loader_ = std::make_unique<StreamLoader>(const_cast<char*>(config_.url.c_str()), stream_id_);

    // 配置拉流器的运行时选项
    StreamLoader::RuntimeOptions loader_options;
    loader_options.max_ready_depth = static_cast<size_t>(std::max(1, config_.loader_max_ready_depth)); // 就绪队列深度
    loader_options.rtbufsize_bytes = config_.loader_rtbufsize_bytes;   // FFmpeg 接收缓冲大小
    loader_options.stimeout_us = config_.loader_stimeout_us;           // 流超时
    loader_options.max_delay_us = config_.loader_max_delay_us;         // 最大延迟
    loader_options.rtsp_transport = config_.loader_rtsp_transport;     // RTSP 传输协议
    loader_->configureRuntime(loader_options); // 应用运行时配置

    // 步骤 5: 激活执行线程
    running_ = true; // 设置运行标志（必须在创建线程之前）

    // ---- 线程池 A: 负责底层码流拉取与 MPP 硬件解码（生产者） ----
    loader_thread_ = std::thread([this]() {
        // 设置线程运行时属性：CPU 亲缘性绑定 + 调度策略
        utils::applyThreadRuntime("rtsp_loader", "rtsp-loader");
        if (loader_) {
            // open/read/reconnect 全部在本线程执行。TCP listener 等待 Windows
            // 连接时不会阻塞 IMX415、Mosaic 或主程序生命周期。
            (*loader_)();
        }
    });

    // ---- 线程池 B: 负责主业务逻辑（推理、RGA转换、后处理、推流、归档）（消费者） ----
    processing_thread_ = std::thread([this]() {
        // 设置线程运行时属性：CPU 亲缘性绑定到 "rtsp_process" 组
        utils::applyThreadRuntime("rtsp_process", "rtsp-process");
        processingLoop(); // 进入主处理循环
    });

    // ---- 线程池 C: 负责异步 NPU 推理（可选） ----
    if (config_.inference.enable && inference_service_) {
        inference_thread_ = std::thread(&VisibleRtspPipeline::inferenceLoop, this);
    }

    std::cout << "[VisRtspPipeline] Pipeline (ID:" << stream_id_ << ") started successfully" << std::endl;
    return true;
}

/**
 * @brief 停止可见光 RTSP 处理管道
 *
 * 停止顺序遵循"先打停止标记，再停上游生产者，再等消费者退出"的原则，
 * 这样可以避免 processingLoop 卡在等待新帧，而 StreamLoader 已经被提前析构。
 */
void VisibleRtspPipeline::stop() {
    if (!running_) {
        return; // 已在停止状态
    }

    // ---- 第一步：设置停止标志 ----
    // 停止顺序遵循"先打停止标记，再停上游生产者，再等消费者退出"的原则，
    // 这样可以避免 processingLoop 卡在等待新帧，而 StreamLoader 已经被提前析构。
    running_ = false;          // 原子标志：通知所有线程退出
    inference_cv_.notify_all(); // 唤醒推理线程（如果正在等待）

    // ---- 第二步：停止生产者 ----
    if (loader_) {
        loader_->stopFlag = true; // 设置 StreamLoader 的停止标志
    }

    // ---- 第三步：等待所有线程退出 ----
    if (loader_thread_.joinable()) {
        loader_thread_.join(); // 等待拉流/解码线程
    }
    if (processing_thread_.joinable()) {
        processing_thread_.join(); // 等待处理线程
    }
    if (inference_thread_.joinable()) {
        inference_thread_.join(); // 等待推理线程
    }

    // ---- 第四步：清理推理残余状态 ----
    {
        std::lock_guard<std::mutex> lock(inference_mutex_);
        inference_pending_ = false;    // 清除待处理标志
        pending_inference_ = {};       // 清空待推理任务
    }

    // ---- 第五步：释放拉流器 ----
    loader_.reset();
}

/**
 * @brief 处理循环，从流加载器获取帧并进行处理
 *
 * 该函数是管道的核心消费者线程逻辑，负责：
 * 1. 从硬件解码器（StreamLoader）获取 DMA-BUF FD 帧描述符
 * 2. 通过 RGA 硬件加速器进行按需格式转换（NV12 -> BGR888）
 * 3. 调度 NPU 推理服务（支持 FD 直连推理，零拷贝）
 * 4. 将处理后的帧通过 frame_callback_ 传给 FrameHub
 * 5. 附带上检测缓存结果、帧率统计等元数据
 *
 * 性能优化：
 * - 使用静态 thread_local RGA 转换器，避免每帧重建
 * - 推理使用独立线程异步执行，不阻塞主处理循环
 * - 转换后的 BGR 帧显示先完成，推理可复用硬件帧 FD
 */
void VisibleRtspPipeline::processingLoop() {
    auto last_submit = std::chrono::steady_clock::time_point{}; // 上次提交推理的时间
    int64_t pts_anchor_us = -1;
    int64_t mono_anchor_ns = 0;
    int64_t last_pts_us = -1;
    while (running_) {
        // ---- 检查拉流器是否有效 ----
        if (!loader_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); // 等待拉流器初始化
            continue;
        }

        // ---- 从解码队列获取一帧 ----
        uint64_t desc_id = 0;                          // 帧描述符 ID
        StreamLoader::DecodedHwFrameDesc hw_desc{};    // 硬件帧描述符（含 DMA-BUF FD）
        {
            ScopedTimer capture_timer(metrics_.last_capture_us); // 计时：采集耗时
            const int wait_ms = config_.pull_interval_ms > 0 ? config_.pull_interval_ms : 10;
            // 等待并获取解码帧（带超时）
            if (!loader_->waitAndGetDecodedFrame(
                    desc_id, hw_desc, std::chrono::milliseconds(wait_ms))) {
                // 超时未获取到帧，更新队列深度统计后继续循环
                metrics_.queue_depth.store(static_cast<uint32_t>(loader_->readyDepth()));
                metrics_.queue_drops.store(loader_->droppedTotalCount());
                continue;
            }
        }

        // RAII 守卫：离开作用域时自动归还帧引用
        DecodedFrameGuard guard{loader_.get(), desc_id};

        // 更新采集统计
        metrics_.frames_in.fetch_add(1, std::memory_order_relaxed); // 累计输入帧数
        metrics_.queue_depth.store(static_cast<uint32_t>(loader_->readyDepth())); // 队列深度
        metrics_.queue_drops.store(loader_->droppedTotalCount());  // 累计丢弃
        ScopedTimer total_timer(metrics_.last_total_us);           // 计时：处理总耗时

        // ---- 记录帧时间戳 ----
        auto now = std::chrono::steady_clock::now();                                // 单调时钟（用于计算间隔）
        auto timestamp = std::chrono::system_clock::now();                          // 系统时钟（对外时间戳）
        const int64_t now_mono_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
        int64_t capture_mono_ns = now_mono_ns;
        if (hw_desc.pts_us >= 0) {
            const bool discontinuity = pts_anchor_us < 0 ||
                (last_pts_us >= 0 &&
                 (hw_desc.pts_us < last_pts_us || hw_desc.pts_us - last_pts_us > 1000000));
            if (discontinuity) {
                pts_anchor_us = hw_desc.pts_us;
                mono_anchor_ns = now_mono_ns;
            }
            int64_t mapped_ns = mono_anchor_ns + (hw_desc.pts_us - pts_anchor_us) * 1000LL;
            // WSL与板端单调时钟会有微小频差。超出可控窗口时立即重新锚定；
            // 同批提前解出的下一帧则按PTS等待到应有时刻再发布，消除MPP突发而不丢帧。
            if (mapped_ns > now_mono_ns + 60000000LL ||
                now_mono_ns - mapped_ns > 80000000LL) {
                pts_anchor_us = hw_desc.pts_us;
                mono_anchor_ns = now_mono_ns;
                mapped_ns = mono_anchor_ns;
            } else if (mapped_ns > now_mono_ns) {
                std::this_thread::sleep_until(std::chrono::steady_clock::time_point(
                    std::chrono::nanoseconds(mapped_ns)));
                now = std::chrono::steady_clock::now();
                timestamp = std::chrono::system_clock::now();
            }
            capture_mono_ns = mapped_ns;
            last_pts_us = hw_desc.pts_us;
        }

        // ---- 发布原始NV12 DMA帧 ----
        // 预览/拼接主链不再物化整帧BGR。frame_hold同时被FrameHub和可选推理任务
        // 持有，最后一个消费者释放后MPP才可回收该解码缓冲。
        FrameHub::DmaFrame dma_frame;
        dma_frame.fd = hw_desc.dma_fd;
        dma_frame.width = hw_desc.width;
        dma_frame.height = hw_desc.height;
        dma_frame.width_stride = hw_desc.width_stride;
        dma_frame.height_stride = hw_desc.height_stride;
        dma_frame.buffer_size = hw_desc.buffer_size;
        dma_frame.format = hw_desc.format == MPP_FMT_YUV422SP
            ? DmaPixelFormat::NV16 : DmaPixelFormat::NV12;
        dma_frame.color_space = DmaColorSpace::Bt709Limited;
        dma_frame.lease = hw_desc.frame_hold;
        if (!dma_frame.valid()) {
            metrics_.frames_dropped.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // ---- 调度异步 NPU 推理 ----
        // 显示转换结束后再把硬件帧引用交给异步推理，避免 RGA 与 MPP 回收并发访问。
        // 按 inference_interval_ms 间隔提交推理任务。
        if (config_.inference.enable && inference_service_ && hw_desc.valid() &&
            (last_submit.time_since_epoch().count() == 0 ||
             std::chrono::duration_cast<std::chrono::milliseconds>(now - last_submit).count() >=
                 config_.inference.interval_ms)) {
            {
                std::lock_guard<std::mutex> lock(inference_mutex_);
                // frame_hold 共享令牌独立保活 DMA 缓冲；覆盖旧任务时自动释放旧令牌
                pending_inference_ = InferenceTask{hw_desc, timestamp, capture_mono_ns}; // 打包推理任务
                inference_pending_ = true; // 设置待处理标志
            }
            inference_cv_.notify_one(); // 唤醒推理线程
            last_submit = now;          // 更新上次提交时间
        }

        // ---- 更新帧率统计与元数据 ----
        frame_rate_.tick();                // 帧率计数器滴答
        metrics_.tickFrame();              // 管线指标更新
        FrameHub::FrameOverlay overlay;
        overlay.source_width = dma_frame.logicalWidth();
        overlay.source_height = dma_frame.logicalHeight();
        overlay.frame_fps = frame_rate_.rate();       // 采集帧率
        overlay.inference_fps = inference_rate_.rate(); // 推理帧率
        overlay.frames_captured = metrics_.frames_in.load(std::memory_order_relaxed);      // 累计采集帧数
        overlay.frames_dropped = metrics_.frames_dropped.load(std::memory_order_relaxed) +
                                 loader_->droppedTotalCount();                             // 累计丢弃帧数

        // ---- 融合检测缓存结果（最近 kMaxReuseDetectionMs 毫秒内的检测结果） ----
        {
            std::lock_guard<std::mutex> lock(detection_mutex_);
            if (detection_cache_valid_) {
                for (size_t i = 0; i < cached_detections_.size(); ++i) {
                    const int64_t detected_at = cached_detection_mono_ns_[i]; // 检测时间戳
                    // 仅使用足够新鲜的检测结果（在复用窗口内）
                    const bool fresh = detected_at > 0 && capture_mono_ns >= detected_at &&
                        capture_mono_ns - detected_at <= kMaxReuseDetectionMs * 1000000LL;
                    if (!fresh) continue; // 太旧的结果跳过
                    overlay.detections[i] = cached_detections_[i]; // 复制检测结果
                    overlay.detections_valid = overlay.detections_valid ||
                                               cached_detections_[i].count > 0; // 更新有效标志
                    overlay.detection_mono_ns = std::max(
                        overlay.detection_mono_ns, detected_at); // 取最新的检测时间
                }
            }
        }

        // ---- 回调：将处理后的帧传给 FrameHub ----
        if (frame_callback_) frame_callback_(dma_frame, timestamp, capture_mono_ns, overlay);
    }
}

/**
 * @brief 推理循环（运行在 inference_thread_ 中）
 *
 * 独立的推理线程，通过条件变量与处理线程通信。
 * 主要流程：
 * 1. 等待 processingLoop 设置 pending_inference_
 * 2. 调用 inference_service_->inferFromFd() 执行 FD 直连推理（零拷贝）
 * 3. 稳定化检测结果（detection_stabilizer）
 * 4. 将结果写入 detection_cache_ 供处理循环复用
 * 5. 调用 inference_callback_ 上报推理统计
 */
void VisibleRtspPipeline::inferenceLoop() {
    // 设置线程运行时属性
    utils::applyThreadRuntime("rtsp_inference", "rtsp-infer");
    while (running_.load()) {
        // ---- 等待推理任务 ----
        InferenceTask task;
        {
            std::unique_lock<std::mutex> lock(inference_mutex_);
            // 等待条件变量：有新任务或停止标志
            inference_cv_.wait_for(lock, std::chrono::milliseconds(200), [this] {
                return !running_.load() || inference_pending_;
            });
            if (!running_.load() && !inference_pending_) break; // 停止且无待处理任务，退出
            if (!inference_pending_) continue;                   // 虚假唤醒，继续等待
            // 取出任务（移动语义，避免拷贝）
            task = std::move(pending_inference_);
            pending_inference_ = {};
            inference_pending_ = false; // 清除待处理标志
        }

        // ---- 执行推理 ----
        const auto started = std::chrono::steady_clock::now();
        InferenceResult result;
        if (inference_service_ && task.desc.valid()) {
            // FD 直连推理：传入 DMA-BUF 文件描述符，零拷贝
            InferenceFrameDesc desc;
            desc.dma_fd = task.desc.dma_fd;
            desc.width = task.desc.width;
            desc.height = task.desc.height;
            desc.width_stride = task.desc.width_stride;
            desc.height_stride = task.desc.height_stride;
            desc.buffer_size = task.desc.buffer_size;
            desc.pixel_format = task.desc.format == MPP_FMT_YUV422SP
                ? RgaPixelFormat::NV16 : RgaPixelFormat::NV12;
            desc.scheduler_slot = 0;
            desc.frame_hold = std::move(task.desc.frame_hold);
            result = inference_service_->inferFromFd(std::move(desc));
        }
        // 记录推理耗时（微秒）
        metrics_.last_inference_us.store(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - started).count());
        if (!result.success) continue; // 推理失败，跳过

        // ---- 稳定化检测结果并写入缓存 ----
        std::array<detect_result_group_t, kMaxInferenceModels> detections{};
        {
            std::lock_guard<std::mutex> lock(detection_mutex_);
            for (size_t i = 0; i < cached_detections_.size(); ++i) {
                if (result.updated_mask & (1u << i)) { // 该模型有更新
                    // 稳定化检测结果（去抖、滤波）
                    stabilizeDetections(cached_detections_[i], result.detections[i]);
                    cached_detections_[i] = result.detections[i];         // 更新缓存
                    cached_detection_mono_ns_[i] = task.capture_mono_ns;  // 记录检测时间
                }
            }
            detections = cached_detections_; // 拷贝到本地（用于回调）
            detection_cache_valid_ = true;   // 标记缓存有效
        }
        inference_rate_.tick(); // 推理帧率计数

        // ---- 调用推理回调 ----
        if (inference_callback_) {
            data_lifecycle::InferenceStats stats;
            stats.stream_id = stream_id_;                       // 流 ID
            stats.person_count = detections[0].count;           // 人数（模型 0）
            stats.ppe_count = detections[2].count;              // PPE 数量（模型 2）
            stats.ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                task.timestamp.time_since_epoch()).count();     // 时间戳（毫秒）
            stats.capture_mono_ns = task.capture_mono_ns;       // 采集时间（纳秒）
            stats.infer_executed = true;                        // 推理已执行
            inference_callback_(stats); // 上报统计
        }
    }
}

} // namespace pipeline
