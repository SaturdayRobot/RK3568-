/**
 * @file rga_preprocessor.cpp
 * @brief RGA 硬件加速预处理模块实现
 *
 * 封装 im2d API（RGA2/RGA3），支持以下硬件加速操作：
 * - resize：图像缩放（支持 BGR888/RGB888/NV12/NV21/RGBA/BGRA 格式）
 * - cvtColor：色彩空间转换（NV12↔BGR，RGB↔BGR 等）
 * - rotate：90°/180°/270° 旋转（仅 BGR888）
 * - 组合操作：resize + cvtColor 一站完成
 *
 * 关键设计：
 * - RgaSubmissionGuard：全局串行化 RGA 硬件访问，防止多线程并发提交导致内核竞争
 * - strict_hardware 开关：禁用 OpenCV CPU 回退路径（实时场景强制使用硬件加速）
 * - DMA-BUF FD 支持：从硬件解码器直接导入帧描述符，零拷贝处理
 * - 连续 3 次 RGA 失败自动禁用硬件加速，回退到 CPU 路径
 * - 色彩空间感知：支持 BT.601/BT.709 的 Full Range 和 Limited Range
 */
#include "pipeline/rga_preprocessor.h"

#include <iostream>            // 控制台输出
#include <mutex>               // std::mutex, std::lock_guard
#include <limits>              // std::numeric_limits
#include <condition_variable>  // std::condition_variable

#include "rga.h"               // RGA 底层驱动 API
#include "im2d.h"              // RGA 2D 图像处理高级 API
#include "RgaUtils.h"          // RGA 工具函数

namespace pipeline {
namespace {

// ============================================================================
// RGA 全局调度器（伪票据锁）
//
// 由于 RGA 驱动的 IOCTL 接口不是完全线程安全的（多线程并发提交可能导致
// 内核缓冲区竞争），这里使用票据排队机制将所有 RGA 操作串行化。
//
// 工作原理：
// - 每个 RGA 操作获取一个递增的 ticket 号
// - 等待直到自己的 ticket 被服务（serving_ticket == ticket）
// - 操作完成后推进 serving_ticket 并通知下一个等待者
// ============================================================================
std::mutex g_rga_scheduler_mutex;            // 调度器互斥锁
std::condition_variable g_rga_scheduler_cv;  // 调度器条件变量
uint64_t g_rga_next_ticket = 0;             // 下一个可分配的票据号
uint64_t g_rga_serving_ticket = 0;          // 当前正在服务的票据号

// RGA 运行时健康状态跟踪
std::atomic<int> g_rga_failure_streak{0};          // 连续失败计数
std::atomic<bool> g_rga_runtime_disabled{false};   // 运行时禁用标志
}

// ============================================================================
// RgaSubmissionGuard: 构造函数
// 获取票据号并排队等待，确保独占 RGA 硬件访问权
// ============================================================================
RgaSubmissionGuard::RgaSubmissionGuard() {
    std::unique_lock<std::mutex> lock(g_rga_scheduler_mutex);
    ticket_ = g_rga_next_ticket++; // 获取当前票据号并递增下一个
    // 等待直到轮到自己的票据号（保证 FIFO 顺序）
    g_rga_scheduler_cv.wait(lock, [this] {
        return ticket_ == g_rga_serving_ticket;
    });
}

// ============================================================================
// RgaSubmissionGuard: 析构函数
// 推进服务票据号并通知下一个等待者
// ============================================================================
RgaSubmissionGuard::~RgaSubmissionGuard() {
    {
        std::lock_guard<std::mutex> lock(g_rga_scheduler_mutex);
        ++g_rga_serving_ticket; // 推进到下一个票据
    }
    // notify_all 是必要的：notify_one 可能唤醒非当前 ticket 并造成无人继续推进
    g_rga_scheduler_cv.notify_all();
}

// rgaRuntimeHealthy: 查询 RGA 运行时是否健康（未被连续故障禁用）
bool rgaRuntimeHealthy() {
    return !g_rga_runtime_disabled.load(std::memory_order_relaxed);
}

// ============================================================================
// reportRgaResult: 上报 RGA 操作结果
// @param success true 表示操作成功
//
// 连续 3 次失败后自动禁用 RGA 硬件加速（回退到 CPU 路径），
// 成功则重置失败计数。
// ============================================================================
void reportRgaResult(bool success) {
    if (success) {
        g_rga_failure_streak.store(0, std::memory_order_relaxed); // 成功：重置失败计数
        return;
    }
    // 失败：原子递增连续失败计数
    const int failures = g_rga_failure_streak.fetch_add(1, std::memory_order_relaxed) + 1;
    if (failures >= 3 && !g_rga_runtime_disabled.exchange(true)) {
        // 连续 3 次失败，且尚未禁用 → 禁用 RGA 硬件加速
        std::cerr << "[RGA] disabled after repeated failures; further hardware submissions blocked\n";
    }
}
}  // namespace pipeline

#include <opencv2/imgproc.hpp> // OpenCV 图像处理（CPU 回退路径）

namespace pipeline {

namespace {

// ============================================================================
// toRgaFormat: 将 RgaPixelFormat 枚举映射到 im2d RGA_PIXEL_FORMAT 常量
// @param fmt 内部像素格式枚举
// @return    RGA 驱动格式常量（RK_FORMAT_xxx）
// ============================================================================
int toRgaFormat(RgaPixelFormat fmt) {
    switch (fmt) {
        case RgaPixelFormat::BGR888:   return RK_FORMAT_BGR_888;       // BGR 8-8-8 三通道
        case RgaPixelFormat::RGB888:   return RK_FORMAT_RGB_888;       // RGB 8-8-8 三通道
        case RgaPixelFormat::NV12:     return RK_FORMAT_YCbCr_420_SP;  // YUV 4:2:0 半平面（Y + UV 交错）
        case RgaPixelFormat::NV21:     return RK_FORMAT_YCrCb_420_SP;  // YUV 4:2:0 半平面（Y + VU 交错）
        case RgaPixelFormat::NV16:     return RK_FORMAT_YCbCr_422_SP;  // YUV 4:2:2 半平面（Y + UV 交错）
        case RgaPixelFormat::RGBA8888: return RK_FORMAT_RGBA_8888;     // RGBA 8-8-8-8 四通道
        case RgaPixelFormat::BGRA8888: return RK_FORMAT_BGRA_8888;     // BGRA 8-8-8-8 四通道
        default:                       return RK_FORMAT_BGR_888;       // 默认回退 BGR888
    }
}

// ============================================================================
// probeRga: 检测 RGA 硬件是否可用
// @return true 表示 RGA 驱动可用
//
// 通过查询 RGA 版本字符串判断是否可用，避免 "恒 true" 误判。
// 即使设备节点存在，也可能因为内核模块未加载而不可用。
// ============================================================================
bool probeRga() {
    // 通过查询 RGA 版本字符串判断是否可用，避免 "恒 true" 误判
    const char* ver = querystring(RGA_VERSION);
    return (ver != nullptr && ver[0] != '\0'); // 非空版本号表示可用
}

// ============================================================================
// cvtColorCode: 获取 OpenCV 色彩空间转换代码
// @param src 源像素格式
// @param dst 目标像素格式
// @return    OpenCV cv::COLOR_xxx 代码，-1 表示不需要转换或不支持
// ============================================================================
int cvtColorCode(RgaPixelFormat src, RgaPixelFormat dst) {
    if (src == RgaPixelFormat::BGR888 && dst == RgaPixelFormat::RGB888) {
        return cv::COLOR_BGR2RGB;                           // BGR → RGB
    }
    if (src == RgaPixelFormat::RGB888 && dst == RgaPixelFormat::BGR888) {
        return cv::COLOR_RGB2BGR;                           // RGB → BGR
    }
    if (src == RgaPixelFormat::NV12 && dst == RgaPixelFormat::BGR888) {
        return cv::COLOR_YUV2BGR_NV12;                      // NV12 → BGR（最常用：解码输出 → 显示）
    }
    if (src == RgaPixelFormat::NV12 && dst == RgaPixelFormat::RGB888) {
        return cv::COLOR_YUV2RGB_NV12;                      // NV12 → RGB
    }
    if (src == RgaPixelFormat::NV21 && dst == RgaPixelFormat::BGR888) {
        return cv::COLOR_YUV2BGR_NV21;                      // NV21 → BGR
    }
    if (src == RgaPixelFormat::NV21 && dst == RgaPixelFormat::RGB888) {
        return cv::COLOR_YUV2RGB_NV21;                      // NV21 → RGB
    }
    return -1; // 不需要转换或不支持的转换对
}

} // unnamed namespace

// ============================================================================
// 构造/析构（使用默认实现）
// ============================================================================
RgaPreprocessor::RgaPreprocessor() = default;
RgaPreprocessor::~RgaPreprocessor() = default;

// ============================================================================
// initialize: 初始化 RGA 预处理器
// @param config RGA 预处理配置
// @return      true 表示初始化成功（即使 RGA 不可用，配置也会被保存）
//
// 初始化流程：
// 1. 保存配置
// 2. 如果启用了 RGA，探测硬件可用性
// 3. 根据 strict_hardware 决定是否允许 CPU 回退
// ============================================================================
bool RgaPreprocessor::initialize(const RgaPreprocessConfig& config) {
    config_ = config;           // 保存配置
    rga_available_ = false;     // 默认不可用

    if (config_.use_rga) {
        try {
            rga_available_ = probeRga(); // 探测 RGA 硬件
            if (rga_available_) {
                std::cerr << "[RgaPreprocessor] RGA hardware available" << std::endl;
            }
        } catch (...) {
            rga_available_ = false; // 探测异常
        }
    }

    if (!rga_available_ && config_.use_rga) {
        if (config_.strict_hardware) {
            // 严格模式：RGA 不可用则报错
            std::cerr << "[RgaPreprocessor] RGA not available (strict hardware mode)" << std::endl;
        } else {
            // 非严格模式：自动回退 CPU
            std::cerr << "[RgaPreprocessor] RGA not available, falling back to CPU" << std::endl;
        }
    }

    initialized_ = true; // 标记已初始化
    return true;
}

// ============================================================================
// process: 主处理入口（cv::Mat 输入 → cv::Mat 输出）
// @param src 输入图像
// @param dst 输出图像（由函数内部分配）
// @return    true 表示处理成功
//
// 处理策略：
// 1. 优先使用 RGA 硬件加速
// 2. RGA 失败且非严格模式 → 回退到 OpenCV CPU 路径
// 3. 严格模式下 RGA 失败 → 直接返回 false
// ============================================================================
bool RgaPreprocessor::process(const cv::Mat& src, cv::Mat& dst) {
    if (!initialized_) {
        return false; // 未初始化
    }
    if (src.empty()) {
        return false; // 空输入
    }

    // 尝试 RGA 硬件路径
    if (rga_available_) {
        if (processRga(src, dst)) {
            return true; // RGA 成功
        }
        // RGA 失败
        if (config_.strict_hardware) {
            std::cerr << "[RgaPreprocessor] RGA process failed (strict hardware mode)" << std::endl;
            return false; // 严格模式不降级
        }
        // RGA 调用失败，回退 CPU
        std::cerr << "[RgaPreprocessor] RGA process failed, falling back to CPU" << std::endl;
    }

    if (config_.strict_hardware) {
        return false; // 严格模式且 RGA 不可用
    }

    // CPU 回退路径
    return processCpu(src, dst);
}

// ============================================================================
// processToBuffer: 处理到外部预分配缓冲区（而非 cv::Mat）
// @param src              输入图像
// @param dst_data         目标缓冲区指针
// @param dst_stride       目标行步长（字节）
// @param dst_height_stride 目标列步长
// @return                 true 表示成功
//
// 用于将结果直接写入外部管理的 buffer，避免额外拷贝。
// 注意：此方法不支持 CPU 回退。
// ============================================================================
bool RgaPreprocessor::processToBuffer(const cv::Mat& src, uint8_t* dst_data, int dst_stride,
                                      int dst_height_stride) {
    if (!initialized_ || src.empty() || !dst_data || dst_stride <= 0) {
        return false; // 参数无效
    }

    if (!rga_available_) {
        return false; // RGA 不可用时不支持（无 CPU 回退）
    }

    return processRgaToBuffer(src, dst_data, dst_stride, dst_height_stride);
}

// ============================================================================
// processFromFd: 从 DMA-BUF 文件描述符处理（零拷贝路径）
// @param src_fd            源帧的 DMA-BUF 文件描述符
// @param src_width         源帧宽度
// @param src_height        源帧高度
// @param src_stride        源帧行步长（像素），0 表示等于宽度
// @param dst               输出 BGR 图像
// @param src_height_stride 源帧列步长，0 表示等于高度
// @param src_buffer_size   DMA-BUF 缓冲区大小（字节），0 表示自动计算
// @return                  true 表示成功
//
// 这是最重要的处理路径之一：直接从 MPP 硬件解码器获取 DMA-BUF FD，
// 通过 RGA 直接处理，无需 CPU 拷贝中间数据。实现真正的零拷贝管线。
// 注意：此方法不支持 CPU 回退（没有 CPU 能直接从 FD 读取 NV12）。
// ============================================================================
bool RgaPreprocessor::processFromFd(int src_fd, int src_width, int src_height,
                                     int src_stride, cv::Mat& dst,
                                     int src_height_stride, size_t src_buffer_size) {
    if (!initialized_) {
        return false; // 未初始化
    }

    if (rga_available_ && src_fd >= 0) {
        // RGA 硬件路径：导入 FD 并处理
        return processRgaFd(src_fd, src_width, src_height, src_stride, dst,
                            src_height_stride, src_buffer_size);
    }

    std::cerr << "[RgaPreprocessor] processFromFd: RGA not available or invalid fd" << std::endl;
    return false;
}

bool RgaPreprocessor::processFdToFdLetterbox(
        int src_fd, int src_width, int src_height,
        int src_stride, int src_height_stride, size_t src_buffer_size,
        RgaPixelFormat src_format,
        int dst_fd, int dst_width, int dst_height,
        int dst_stride, int dst_height_stride, size_t dst_buffer_size,
        int resized_width, int resized_height, int offset_x, int offset_y,
        bool clear_padding,
        uint8_t padding_value) {
    if (!initialized_ || !rga_available_ || src_fd < 0 || dst_fd < 0) {
        return false;
    }
    return processRgaFdToFdLetterbox(
        src_fd, src_width, src_height, src_stride, src_height_stride, src_buffer_size,
        src_format,
        dst_fd, dst_width, dst_height, dst_stride, dst_height_stride, dst_buffer_size,
        resized_width, resized_height, offset_x, offset_y, clear_padding, padding_value);
}

// ============================================================================
// toRgaColorSpace: 将色彩空间枚举映射到 RGA YUV↔RGB 转换模式
// @param color_space 色彩空间枚举（Bt601Full/Bt601Limited/Bt709Limited）
// @param yuv_to_rgb  true 表示 YUV→RGB，false 表示 RGB→YUV
// @return            RGA 色彩空间常量
// ============================================================================
int toRgaColorSpace(RgaColorSpace color_space, bool yuv_to_rgb) {
    if (yuv_to_rgb) {
        // YUV → RGB 转换
        if (color_space == RgaColorSpace::Bt601Full) return IM_YUV_TO_RGB_BT601_FULL;     // BT.601 Full Range
        if (color_space == RgaColorSpace::Bt601Limited) return IM_YUV_TO_RGB_BT601_LIMIT; // BT.601 Limited (SD)
        return IM_YUV_TO_RGB_BT709_LIMIT; // 默认 BT.709 Limited (HD)
    }
    // RGB → YUV 转换
    if (color_space == RgaColorSpace::Bt601Full) return IM_RGB_TO_YUV_BT601_FULL;
    if (color_space == RgaColorSpace::Bt601Limited) return IM_RGB_TO_YUV_BT601_LIMIT;
    return IM_RGB_TO_YUV_BT709_LIMIT; // 默认 BT.709
}

// ============================================================================
// toRgaFullColorSpace: 将色彩空间枚举映射到 RGA YUV Range 模式
// @param color_space 色彩空间枚举
// @return            RGA Color Space Mode 常量
//
// 用于设置 rga_buffer_t::color_space_mode，通知 RGA 驱动输入/输出的
// YUV 数据范围（Full Range: 0-255, Limited Range: 16-235）。
// ============================================================================
int toRgaFullColorSpace(RgaColorSpace color_space) {
    if (color_space == RgaColorSpace::Bt601Full) return IM_YUV_BT601_FULL_RANGE;     // BT.601 Full Range
    if (color_space == RgaColorSpace::Bt601Limited) return IM_YUV_BT601_LIMIT_RANGE; // BT.601 Limited Range
    return IM_YUV_BT709_LIMIT_RANGE; // 默认 BT.709 Limited Range
}

// ============================================================================
// rotateBgr: RGA 硬件加速旋转 BGR888 图像
// @param src      输入 BGR888 图像
// @param dst      输出图像（由函数分配）
// @param rotation 旋转角度（仅支持 90/180/270）
// @return         true 表示成功
//
// 使用 imrotate API 执行硬件旋转，相比 OpenCV 的 cv::rotate 更高效且
// 不占用 CPU。
// ============================================================================
bool RgaPreprocessor::rotateBgr(const cv::Mat& src, cv::Mat& dst, int rotation) const {
    // 前置检查
    if (!initialized_ || !rga_available_ || !rgaRuntimeHealthy() ||
        src.empty() || src.type() != CV_8UC3) {
        return false;
    }
    RgaSubmissionGuard rga_guard; // 获取 RGA 访问权
    if (!rgaRuntimeHealthy()) return false; // 再次检查健康状态

    // 根据旋转角度确定变换模式和目标尺寸
    int transform = 0;
    int dst_width = src.cols;
    int dst_height = src.rows;
    if (rotation == 90) {
        transform = IM_HAL_TRANSFORM_ROT_90;   // 顺时针 90°
        dst_width = src.rows;                  // 宽高交换
        dst_height = src.cols;
    } else if (rotation == 180) {
        transform = IM_HAL_TRANSFORM_ROT_180;  // 180°（宽高不变）
    } else if (rotation == 270) {
        transform = IM_HAL_TRANSFORM_ROT_270;  // 顺时针 270°（等价于逆时针 90°）
        dst_width = src.rows;
        dst_height = src.cols;
    } else {
        return false; // 不支持的角度
    }

    // ROI/submatrix 的 data 可能位于原始分配中部。RGA 的虚拟地址导入无法从
    // cv::Mat 获知 datastart/dataend，统一复制为独立连续缓冲，避免映射范围越界。
    const cv::Mat safe_src = (!src.isContinuous() || src.data != src.datastart)
                                 ? src.clone()
                                 : src;

    // 分配目标缓冲区
    dst.create(dst_height, dst_width, CV_8UC3);

    // 构建源 RGA buffer（wrapbuffer_virtualaddr 自动计算步长）
    rga_buffer_t input = wrapbuffer_virtualaddr(
        const_cast<uint8_t*>(safe_src.data), safe_src.cols, safe_src.rows, RK_FORMAT_BGR_888);
    input.wstride = static_cast<int>(safe_src.step / safe_src.elemSize()); // 手动设置行步长

    // 构建目标 RGA buffer
    rga_buffer_t output = wrapbuffer_virtualaddr(
        dst.data, dst.cols, dst.rows, RK_FORMAT_BGR_888);
    output.wstride = static_cast<int>(dst.step / dst.elemSize());

    // 执行 RGA 旋转
    const bool success = imrotate(input, output, transform) == IM_STATUS_SUCCESS;
    reportRgaResult(success); // 上报结果
    return success;
}

// ── CPU 回退实现 ──────────────────────────────────────────
// 以下代码仅在 RGA 不可用或失败时执行，作为 OpenCV CPU 回退路径。
// ─────────────────────────────────────────────────────────

// ============================================================================
// processCpu: CPU 回退路径（OpenCV 实现）
// @param src 输入图像
// @param dst 输出图像
// @return    true 表示成功
//
// 处理步骤：
// 1. 色彩空间转换（如果需要）
// 2. 尺寸缩放（如果需要）
// 注意：这两步在某些情况下可以合并（cv::resize 本身不支持 cvtColor 合并）
// ============================================================================
bool RgaPreprocessor::processCpu(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat converted;

    // Step 1: 颜色空间转换
    int code = cvtColorCode(config_.src_format, config_.dst_format); // 获取转换代码
    if (code >= 0) {
        cv::cvtColor(src, converted, code); // OpenCV 色彩转换
    } else {
        converted = src; // 无需转换（格式相同）
    }

    // Step 2: 尺寸缩放
    if (converted.cols != config_.target_width || converted.rows != config_.target_height) {
        cv::resize(converted, dst, cv::Size(config_.target_width, config_.target_height),
                   0, 0, cv::INTER_LINEAR); // 双线性插值缩放
    } else {
        dst = converted; // 尺寸已匹配，直接赋值
    }

    return !dst.empty(); // 返回是否成功
}

// ── RGA 硬件实现 ─────────────────────────────────────────
// 以下代码使用 im2d API 直接调用 RGA 硬件加速器。
// 支持的组合操作：
// - imresize：缩放 + 色彩转换（一站完成）
// - imcvtcolor：仅色彩转换（不改变尺寸时更高效）
// - importbuffer_fd：从 DMA-BUF FD 导入数据（零拷贝）
// ─────────────────────────────────────────────────────────

// ============================================================================
// processRga: RGA 硬件路径（cv::Mat → cv::Mat）
// @param src 输入图像
// @param dst 输出图像（由函数分配）
// @return    true 表示成功
//
// 自动选择最优 RGA 操作：
// - 尺寸不同 → imresize（内置色彩转换）
// - 尺寸相同但格式不同 → imcvtcolor（更高效）
// - 尺寸相同且格式相同 → 直接拷贝（实际上不调用 RGA）
// ============================================================================
bool RgaPreprocessor::processRga(const cv::Mat& src, cv::Mat& dst) {
    if (!rgaRuntimeHealthy()) return false; // 运行时健康检查
    RgaSubmissionGuard rga_guard;           // 获取 RGA 硬件的独占访问权
    if (!rgaRuntimeHealthy()) return false; // 再次检查

    int src_format = toRgaFormat(config_.src_format); // 源 RGA 格式
    int dst_format = toRgaFormat(config_.dst_format); // 目标 RGA 格式

    // ---- 分配目标缓冲区 ----
    // NV12/NV21 为 1.5 字节/像素，单通道连续存放
    if (config_.dst_format == RgaPixelFormat::NV12 ||
        config_.dst_format == RgaPixelFormat::NV21) {
        // NV12: H*1.5 行，宽度=W，单通道（Y 平面 H 行 + UV 平面 H/2 行）
        dst.create(config_.target_height * 3 / 2, config_.target_width, CV_8UC1);
    } else {
        // RGB/BGR/RGBA/BGRA 格式
        int dst_channels = 3; // 默认 3 通道
        if (config_.dst_format == RgaPixelFormat::RGBA8888 ||
            config_.dst_format == RgaPixelFormat::BGRA8888) {
            dst_channels = 4; // RGBA 为 4 通道
        }
        dst.create(config_.target_height, config_.target_width, CV_8UC(dst_channels));
    }

    // 确保源数据连续（RGA 需要连续内存布局）
    const cv::Mat safe_src = (!src.isContinuous() || src.data != src.datastart)
                                 ? src.clone()
                                 : src;

    // ---- 构建 RGA 源/目标缓冲区描述 ----
    rga_buffer_t rga_src = wrapbuffer_virtualaddr(
        const_cast<void*>(static_cast<const void*>(safe_src.data)),
        safe_src.cols, safe_src.rows, src_format);
    rga_src.wstride = static_cast<int>(safe_src.step / safe_src.elemSize()); // 行步长

    rga_buffer_t rga_dst = wrapbuffer_virtualaddr(
        dst.data,
        config_.target_width, config_.target_height, dst_format);

    // ---- 设置色彩空间模式 ----
    const bool src_yuv = config_.src_format == RgaPixelFormat::NV12 ||
                         config_.src_format == RgaPixelFormat::NV21; // 源是否为 YUV
    const bool dst_yuv = config_.dst_format == RgaPixelFormat::NV12 ||
                         config_.dst_format == RgaPixelFormat::NV21; // 目标是否为 YUV
    rga_src.color_space_mode = src_yuv ? toRgaFullColorSpace(config_.color_space) : IM_RGB_FULL;
    rga_dst.color_space_mode = dst_yuv ? toRgaFullColorSpace(config_.color_space) : IM_RGB_FULL;

    // ---- 选择最优 RGA 操作 ----
    // 尺寸相同且格式不同时使用 imcvtcolor（仅色彩转换）；否则 imresize 同时处理 resize+cvtColor
    IM_STATUS status;
    if (src.cols == config_.target_width && src.rows == config_.target_height &&
        src_format != dst_format) {
        // 仅色彩转换（不改变尺寸）
        status = imcvtcolor(rga_src, rga_dst, src_format, dst_format,
                            toRgaColorSpace(config_.color_space, src_yuv));
    } else {
        // 缩放 + 色彩转换（imresize 内置色彩空间转换能力）
        status = imresize(rga_src, rga_dst);
    }
    if (status != IM_STATUS_SUCCESS) {
        reportRgaResult(false); // 记录失败
        std::cerr << "[RgaPreprocessor] RGA operation failed: " << imStrError(status) << std::endl;
        return false;
    }
    reportRgaResult(true); // 记录成功
    return true;
}

// ============================================================================
// processRgaToBuffer: RGA 硬件路径（cv::Mat → 外部预分配缓冲区）
// @param src              输入图像
// @param dst_data         目标缓冲区指针
// @param dst_stride       目标行步长（字节）
// @param dst_height_stride 目标列步长
// @return                 true 表示成功
//
// 与 processRga 的区别：目标不是 cv::Mat 而是外部预分配的 uint8_t* 缓冲区。
// 需要手动校验 stride 的有效性。
// ============================================================================
bool RgaPreprocessor::processRgaToBuffer(const cv::Mat& src, uint8_t* dst_data, int dst_stride,
                                         int dst_height_stride) {
    if (!rgaRuntimeHealthy()) return false; // 运行时健康检查
    RgaSubmissionGuard rga_guard;           // 获取 RGA 硬件的独占访问权
    if (!rgaRuntimeHealthy()) return false; // 再次检查

    int src_format = toRgaFormat(config_.src_format); // 源格式
    int dst_format = toRgaFormat(config_.dst_format); // 目标格式

    // 确保源数据连续
    const cv::Mat safe_src = (!src.isContinuous() || src.data != src.datastart)
                                 ? src.clone()
                                 : src;

    // 构建源 RGA buffer
    rga_buffer_t rga_src = wrapbuffer_virtualaddr(
        const_cast<void*>(static_cast<const void*>(safe_src.data)),
        safe_src.cols, safe_src.rows, src_format);
    rga_src.wstride = static_cast<int>(safe_src.step / safe_src.elemSize());

    // 构建目标 RGA buffer（外部预分配内存）
    rga_buffer_t rga_dst = wrapbuffer_virtualaddr(
        dst_data,
        config_.target_width, config_.target_height, dst_format);

    // 设置列步长（如果指定了 height_stride）
    const int effective_height_stride = dst_height_stride > 0
        ? dst_height_stride : config_.target_height;
    if (effective_height_stride < config_.target_height) return false; // 列步长不足
    rga_dst.hstride = effective_height_stride;

    // 校验行步长：必须 ≥ 目标宽度 × 每像素字节数，且对齐到像素字节
    const int dst_pixel_bytes = (config_.dst_format == RgaPixelFormat::BGR888 ||
                                 config_.dst_format == RgaPixelFormat::RGB888) ? 3 :
                                ((config_.dst_format == RgaPixelFormat::RGBA8888 ||
                                  config_.dst_format == RgaPixelFormat::BGRA8888) ? 4 : 1);
    if (dst_stride < config_.target_width * dst_pixel_bytes ||
        dst_stride % dst_pixel_bytes != 0) {
        return false; // stride 无效
    }
    rga_dst.wstride = dst_stride / dst_pixel_bytes; // 将字节步长转为像素步长

    // 设置色彩空间模式
    const bool src_yuv = config_.src_format == RgaPixelFormat::NV12 ||
                         config_.src_format == RgaPixelFormat::NV21;
    const bool dst_yuv = config_.dst_format == RgaPixelFormat::NV12 ||
                         config_.dst_format == RgaPixelFormat::NV21;
    rga_src.color_space_mode = src_yuv ? toRgaFullColorSpace(config_.color_space) : IM_RGB_FULL;
    rga_dst.color_space_mode = dst_yuv ? toRgaFullColorSpace(config_.color_space) : IM_RGB_FULL;

    // 选择最优操作
    IM_STATUS status;
    if (src.cols == config_.target_width && src.rows == config_.target_height &&
        src_format != dst_format) {
        status = imcvtcolor(rga_src, rga_dst, src_format, dst_format,
                            toRgaColorSpace(config_.color_space, src_yuv));
    } else {
        status = imresize(rga_src, rga_dst);
    }
    if (status != IM_STATUS_SUCCESS) {
        reportRgaResult(false);
        std::cerr << "[RgaPreprocessor] RGA processToBuffer failed: " << imStrError(status) << std::endl;
        return false;
    }
    reportRgaResult(true);
    return true;
}

// ============================================================================
// processRgaFd: RGA 硬件路径（DMA-BUF FD → cv::Mat）
// @param src_fd            源 DMA-BUF 文件描述符（来自 MPP 解码器）
// @param src_width         源帧宽度（像素）
// @param src_height        源帧高度（像素）
// @param src_stride        源帧行步长（像素），0 表示等于宽度
// @param dst               输出 BGR/RGB 图像（由函数分配）
// @param src_height_stride 源帧列步长，0 表示等于高度
// @param src_buffer_size   DMA-BUF 缓冲区大小（字节），0 表示自动计算
// @return                  true 表示成功
//
// 这是零拷贝处理的核心路径：
// 1. importbuffer_fd() 直接从内核导入 DMA-BUF FD（无需 CPU 拷贝）
// 2. RGA 硬件直接从 DMA-BUF 读取 NV12 数据
// 3. 执行 resize + cvtColor → BGR888/RGB888
// 4. 目标 cv::Mat 也通过 importbuffer_virtualaddr 转为 handle
//
// 关键约束（librga 1.10.x）：
// - 不允许一个操作中混用 handle buffer 和裸 fd/虚拟地址 buffer
// - 因此源和目标都必须先通过 import 接口转为 handle
// ============================================================================
bool RgaPreprocessor::processRgaFd(int src_fd, int src_width, int src_height,
                                    int src_stride, cv::Mat& dst,
                                    int src_height_stride, size_t src_buffer_size) {
    if (!rgaRuntimeHealthy()) return false; // 运行时健康检查
    RgaSubmissionGuard rga_guard;           // 获取 RGA 硬件的独占访问权
    if (!rgaRuntimeHealthy()) return false; // 再次检查

    int src_format = toRgaFormat(config_.src_format); // 源格式（通常为 NV12）
    int dst_format = toRgaFormat(config_.dst_format); // 目标格式（通常为 BGR888）

    // ---- 分配目标 cv::Mat ----
    int dst_channels = 3; // 默认 3 通道
    if (config_.dst_format == RgaPixelFormat::RGBA8888 ||
        config_.dst_format == RgaPixelFormat::BGRA8888) {
        dst_channels = 4; // 4 通道
    }
    dst.create(config_.target_height, config_.target_width, CV_8UC(dst_channels));

    // ---- 计算有效的 stride 和缓冲区大小 ----
    // 按真实 stride 导入 DMA-BUF。先按 width/height 导入再扩大 wstride 会让 RGA 认为
    // 可访问范围大于已映射范围，在 MPP 对齐 stride 场景可能触发内核越界。
    const int effective_stride = src_stride > 0 ? src_stride : src_width; // MPP 可能使用对齐后的 stride
    const int effective_height_stride = src_height_stride > 0 ? src_height_stride : src_height;

    // 参数合法性校验
    if (effective_stride < src_width || effective_height_stride < src_height ||
        src_height <= 0 || effective_stride > 8192 || effective_height_stride > 8192) {
        return false;
    }

    // 计算 NV12 缓冲区最小所需大小：stride * height * 1.5（Y + UV 半平面）
    const int64_t minimum_size = static_cast<int64_t>(effective_stride) *
                                 effective_height_stride * 3 / 2;
    const int64_t source_size_64 = src_buffer_size > 0
        ? static_cast<int64_t>(src_buffer_size) : minimum_size;
    if (source_size_64 < minimum_size || source_size_64 > std::numeric_limits<int>::max()) {
        return false;
    }

    // ---- 导入源 DMA-BUF FD 为 RGA handle ----
    rga_buffer_handle_t src_handle = importbuffer_fd(src_fd, static_cast<int>(source_size_64));
    if (src_handle == 0) {
        reportRgaResult(false);
        std::cerr << "[RgaPreprocessor] importbuffer_fd failed" << std::endl;
        return false;
    }

    // 从 handle 构建 RGA buffer 描述（使用实际的 width/height，不是 stride）
    rga_buffer_t rga_src = wrapbuffer_handle(src_handle, src_width, src_height, src_format);
    rga_src.wstride = effective_stride;     // 真实行步长（可能大于 width）
    rga_src.hstride = effective_height_stride; // 真实列步长

    // ---- 导入目标虚拟地址为 RGA handle ----
    // librga 1.10.x 不允许一个操作中混用 handle buffer 和裸 fd/虚拟地址
    // buffer。源端已通过 importbuffer_fd() 得到 handle，因此目标虚拟内存也
    // 必须先导入为 handle。
    rga_buffer_handle_t dst_handle = importbuffer_virtualaddr(
        dst.data, config_.target_width, config_.target_height, dst_format);
    if (dst_handle == 0) {
        reportRgaResult(false);
        std::cerr << "[RgaPreprocessor] importbuffer_virtualaddr(dst) failed" << std::endl;
        releasebuffer_handle(src_handle); // 释放源 handle（避免泄漏）
        return false;
    }

    // 从 handle 构建目标 RGA buffer
    rga_buffer_t rga_dst = wrapbuffer_handle(
        dst_handle, config_.target_width, config_.target_height, dst_format);

    // ---- 设置色彩空间模式 ----
    const bool src_yuv = config_.src_format == RgaPixelFormat::NV12 ||
                         config_.src_format == RgaPixelFormat::NV21;
    const bool dst_yuv = config_.dst_format == RgaPixelFormat::NV12 ||
                         config_.dst_format == RgaPixelFormat::NV21;
    rga_src.color_space_mode = src_yuv ? toRgaFullColorSpace(config_.color_space) : IM_RGB_FULL;
    rga_dst.color_space_mode = dst_yuv ? toRgaFullColorSpace(config_.color_space) : IM_RGB_FULL;

    // ---- 选择最优 RGA 操作 ----
    IM_STATUS status;
    if (src_width == config_.target_width && src_height == config_.target_height &&
        src_format != dst_format) {
        // 尺寸相同：仅色彩转换（更高效）
        status = imcvtcolor(rga_src, rga_dst, src_format, dst_format,
                            toRgaColorSpace(config_.color_space, src_yuv));
    } else {
        // 尺寸不同：缩放 + 色彩转换
        status = imresize(rga_src, rga_dst);
    }

    // ---- 释放导入的 handle ----
    // 两端导入的 handle 必须成对释放；cv::Mat 像素内存仍由 OpenCV 管理。
    releasebuffer_handle(dst_handle); // 先释放目标 handle
    releasebuffer_handle(src_handle); // 再释放源 handle    // 注意：释放顺序是先 dst 后 src

    if (status != IM_STATUS_SUCCESS) {
        reportRgaResult(false);
        std::cerr << "[RgaPreprocessor] imresize (fd) failed: " << imStrError(status) << std::endl;
        return false;
    }
    reportRgaResult(true);
    return true;
}

bool RgaPreprocessor::processRgaFdToFdLetterbox(
        int src_fd, int src_width, int src_height,
        int src_stride, int src_height_stride, size_t src_buffer_size,
        RgaPixelFormat src_format,
        int dst_fd, int dst_width, int dst_height,
        int dst_stride, int dst_height_stride, size_t dst_buffer_size,
        int resized_width, int resized_height, int offset_x, int offset_y,
        bool clear_padding,
        uint8_t padding_value) {
    if (!rgaRuntimeHealthy()) return false;
    if ((src_format != RgaPixelFormat::NV12 && src_format != RgaPixelFormat::NV16) ||
        config_.dst_format != RgaPixelFormat::RGB888 ||
        dst_width != config_.target_width || dst_height != config_.target_height ||
        src_width <= 0 || src_height <= 0 || dst_width <= 0 || dst_height <= 0 ||
        resized_width <= 0 || resized_height <= 0 || offset_x < 0 || offset_y < 0 ||
        offset_x + resized_width > dst_width || offset_y + resized_height > dst_height) {
        return false;
    }

    const int source_wstride = src_stride > 0 ? src_stride : src_width;
    const int source_hstride = src_height_stride > 0 ? src_height_stride : src_height;
    const int target_wstride = dst_stride > 0 ? dst_stride : dst_width;
    const int target_hstride = dst_height_stride > 0 ? dst_height_stride : dst_height;
    if (source_wstride < src_width || source_hstride < src_height ||
        target_wstride < dst_width || target_hstride < dst_height ||
        (src_width & 1) != 0 || (src_height & 1) != 0 ||
        (source_wstride & 1) != 0 || (source_hstride & 1) != 0 ||
        source_wstride > 8192 || source_hstride > 8192 ||
        target_wstride > 8192 || target_hstride > 8192) {
        return false;
    }

    const int64_t minimum_source_size = static_cast<int64_t>(source_wstride) *
        source_hstride * (src_format == RgaPixelFormat::NV16 ? 2 : 3) /
        (src_format == RgaPixelFormat::NV16 ? 1 : 2);
    const int64_t source_size = src_buffer_size > 0
        ? static_cast<int64_t>(src_buffer_size) : minimum_source_size;
    const int64_t minimum_target_size = static_cast<int64_t>(target_wstride) *
                                        target_hstride * 3;
    const int64_t target_size = dst_buffer_size > 0
        ? static_cast<int64_t>(dst_buffer_size) : minimum_target_size;
    if (source_size < minimum_source_size || target_size < minimum_target_size ||
        source_size > std::numeric_limits<int>::max() ||
        target_size > std::numeric_limits<int>::max()) {
        return false;
    }

    RgaSubmissionGuard rga_guard;
    if (!rgaRuntimeHealthy()) return false;

    rga_buffer_handle_t src_handle = importbuffer_fd(src_fd, static_cast<int>(source_size));
    if (src_handle == 0) {
        std::cerr << "[RgaPreprocessor] import source DMA-BUF failed" << std::endl;
        return false;
    }
    rga_buffer_handle_t dst_handle = importbuffer_fd(dst_fd, static_cast<int>(target_size));
    if (dst_handle == 0) {
        releasebuffer_handle(src_handle);
        std::cerr << "[RgaPreprocessor] import RKNN input DMA-BUF failed" << std::endl;
        return false;
    }

    rga_buffer_t rga_src = wrapbuffer_handle(
        src_handle, src_width, src_height,
        src_format == RgaPixelFormat::NV16
            ? RK_FORMAT_YCbCr_422_SP : RK_FORMAT_YCbCr_420_SP,
        source_wstride, source_hstride);
    rga_buffer_t rga_dst = wrapbuffer_handle(
        dst_handle, dst_width, dst_height, RK_FORMAT_RGB_888,
        target_wstride, target_hstride);
    rga_src.color_space_mode = toRgaFullColorSpace(config_.color_space);
    rga_dst.color_space_mode = IM_RGB_FULL;

    const im_rect full_target{0, 0, dst_width, dst_height};
    const int padding_color = (static_cast<int>(padding_value) << 16) |
                              (static_cast<int>(padding_value) << 8) |
                              static_cast<int>(padding_value);
    IM_STATUS status = clear_padding
        ? imfill(rga_dst, full_target, padding_color, 1)
        : IM_STATUS_SUCCESS;
    if (status == IM_STATUS_SUCCESS) {
        const im_rect source_rect{0, 0, src_width, src_height};
        const im_rect target_rect{offset_x, offset_y, resized_width, resized_height};
        status = imcheck(rga_src, rga_dst, source_rect, target_rect);
        if (status == IM_STATUS_NOERROR) {
            // IM_SYNC guarantees that the source DMA-BUF can be released on return.
            status = improcess(
                rga_src, rga_dst, {}, source_rect, target_rect, {}, IM_SYNC);
        }
    }

    releasebuffer_handle(dst_handle);
    releasebuffer_handle(src_handle);

    if (status != IM_STATUS_SUCCESS) {
        // This operation may be unsupported by a particular RGA/BSP combination.
        // Do not trip the global RGA circuit breaker and take down display/encode.
        std::cerr << "[RgaPreprocessor] FD-to-RKNN letterbox failed: "
                  << imStrError(status) << std::endl;
        return false;
    }
    reportRgaResult(true);
    return true;
}

bool RgaPreprocessor::composeDmaToFdNv12(
        const std::vector<RgaDmaComposeTask>& tasks,
        int dst_fd, int dst_width, int dst_height,
        int dst_stride, int dst_height_stride, size_t dst_buffer_size) {
    if (!initialized_ || !rga_available_ || !rgaRuntimeHealthy() || tasks.empty() ||
        dst_fd < 0 || dst_width <= 0 || dst_height <= 0) return false;

    const int target_wstride = dst_stride > 0 ? dst_stride : dst_width;
    const int target_hstride = dst_height_stride > 0 ? dst_height_stride : dst_height;
    const int64_t minimum_target_size = static_cast<int64_t>(target_wstride) *
                                        target_hstride * 3 / 2;
    const int64_t target_size = dst_buffer_size > 0
        ? static_cast<int64_t>(dst_buffer_size) : minimum_target_size;
    if (target_wstride < dst_width || target_hstride < dst_height ||
        (target_wstride & 1) || (target_hstride & 1) ||
        target_size < minimum_target_size || target_size > std::numeric_limits<int>::max()) {
        return false;
    }

    RgaSubmissionGuard rga_guard;
    if (!rgaRuntimeHealthy()) return false;

    rga_buffer_handle_t dst_handle = importbuffer_fd(dst_fd, static_cast<int>(target_size));
    if (dst_handle == 0) return false;
    rga_buffer_t rga_dst = wrapbuffer_handle(
        dst_handle, dst_width, dst_height, RK_FORMAT_YCbCr_420_SP,
        target_wstride, target_hstride);
    rga_dst.color_space_mode = IM_YUV_BT709_LIMIT_RANGE;

    // 每帧先清空整张目标画布。否则某一路掉线或 contain/cover ROI 改变后，
    // MPP 复用缓冲中会残留上一帧，表现为冻结画面、脏边，甚至误判为源仍在线。
    const im_rect full_target{0, 0, dst_width, dst_height};
    const IM_STATUS clear_status = imfill(rga_dst, full_target, 0x000000, 1);
    if (clear_status != IM_STATUS_SUCCESS) {
        releasebuffer_handle(dst_handle);
        return false;
    }

    std::vector<rga_buffer_handle_t> source_handles;
    source_handles.reserve(tasks.size());
    std::vector<rga_buffer_t> source_buffers;
    std::vector<im_rect> source_rects;
    std::vector<im_rect> target_rects;
    std::vector<int> usages;
    source_buffers.reserve(tasks.size());
    source_rects.reserve(tasks.size());
    target_rects.reserve(tasks.size());
    usages.reserve(tasks.size());
    auto release_handles = [&]() {
        for (auto handle : source_handles) {
            if (handle != 0) releasebuffer_handle(handle);
        }
        releasebuffer_handle(dst_handle);
    };

    im_job_handle_t job = imbeginJob(0);
    if (job == 0) {
        release_handles();
        return false;
    }

    bool inputs_ok = true;
    bool task_api_ok = true;
    for (const auto& task : tasks) {
        const int source_wstride = task.src_width_stride > 0
            ? task.src_width_stride : task.src_width;
        const int source_hstride = task.src_height_stride > 0
            ? task.src_height_stride : task.src_height;
        const int64_t minimum_source_size = static_cast<int64_t>(source_wstride) *
            source_hstride * (task.src_format == RgaPixelFormat::NV16 ? 2 : 3) /
            (task.src_format == RgaPixelFormat::NV16 ? 1 : 2);
        const int64_t source_size = task.src_buffer_size > 0
            ? static_cast<int64_t>(task.src_buffer_size) : minimum_source_size;
        const cv::Rect source_bounds(0, 0, task.src_width, task.src_height);
        const cv::Rect target_bounds(0, 0, dst_width, dst_height);
        if (task.src_fd < 0 || !task.lease || task.src_width <= 0 || task.src_height <= 0 ||
            source_wstride < task.src_width || source_hstride < task.src_height ||
            source_size < minimum_source_size || source_size > std::numeric_limits<int>::max() ||
            task.source_rect.area() <= 0 || (task.source_rect & source_bounds) != task.source_rect ||
            task.destination_rect.area() <= 0 ||
            (task.destination_rect & target_bounds) != task.destination_rect ||
            (task.destination_rect.x & 1) || (task.destination_rect.y & 1) ||
            (task.destination_rect.width & 1) || (task.destination_rect.height & 1)) {
            inputs_ok = false;
            break;
        }

        const int source_format = task.src_format == RgaPixelFormat::NV21
            ? RK_FORMAT_YCrCb_420_SP
            : task.src_format == RgaPixelFormat::NV16
                ? RK_FORMAT_YCbCr_422_SP : RK_FORMAT_YCbCr_420_SP;
        rga_buffer_handle_t src_handle = importbuffer_fd(
            task.src_fd, static_cast<int>(source_size));
        if (src_handle == 0) {
            inputs_ok = false;
            break;
        }
        source_handles.push_back(src_handle);
        rga_buffer_t rga_src = wrapbuffer_handle(
            src_handle, task.src_width, task.src_height, source_format,
            source_wstride, source_hstride);
        rga_src.color_space_mode = toRgaFullColorSpace(task.color_space);

        int usage = IM_SYNC;
        if (task.rotation == 90) usage |= IM_HAL_TRANSFORM_ROT_90;
        else if (task.rotation == 180) usage |= IM_HAL_TRANSFORM_ROT_180;
        else if (task.rotation == 270) usage |= IM_HAL_TRANSFORM_ROT_270;
        else if (task.rotation != 0) {
            inputs_ok = false;
            break;
        }
        const im_rect source_rect{task.source_rect.x, task.source_rect.y,
                                  task.source_rect.width, task.source_rect.height};
        const im_rect target_rect{task.destination_rect.x, task.destination_rect.y,
                                  task.destination_rect.width, task.destination_rect.height};
        source_buffers.push_back(rga_src);
        source_rects.push_back(source_rect);
        target_rects.push_back(target_rect);
        usages.push_back(usage);
        if (task_api_ok && improcessTask(
                job, rga_src, rga_dst, {}, source_rect, target_rect, {},
                nullptr, usage) != IM_STATUS_SUCCESS) {
            task_api_ok = false;
        }
    }

    IM_STATUS status = IM_STATUS_FAILED;
    if (inputs_ok && task_api_ok) status = imendJob(job, IM_SYNC, 0, nullptr);
    else imcancelJob(job);

    // 老BSP可能提供task API头文件但驱动不接受多任务job。逐路同步提交仍然
    // 直接写同一MPP DMA输入，不退回BGR，也不会影响源缓冲释放时机。
    if (inputs_ok && status != IM_STATUS_SUCCESS &&
        source_buffers.size() == tasks.size()) {
        status = IM_STATUS_SUCCESS;
        for (size_t i = 0; i < source_buffers.size(); ++i) {
            const IM_STATUS one = improcess(
                source_buffers[i], rga_dst, {}, source_rects[i], target_rects[i], {}, usages[i]);
            if (one != IM_STATUS_SUCCESS) {
                status = one;
                break;
            }
        }
    }
    release_handles();
    if (!inputs_ok || status != IM_STATUS_SUCCESS) {
        static std::atomic<uint64_t> failure_count{0};
        const uint64_t count = failure_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (count == 1 || count % 100 == 0) {
            std::cerr << "[RgaPreprocessor] direct DMA mosaic failed: "
                      << imStrError(status) << " count=" << count << std::endl;
        }
        return false;
    }
    reportRgaResult(true);
    return true;
}

} // namespace pipeline
