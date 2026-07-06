#pragma once  // 头文件保护宏，防止重复包含

/**
 * @file rga_preprocessor.h
 * @brief RGA 硬件加速预处理模块
 *
 * 封装 RGA im2d API，提供统一的图像 resize / cvtColor / crop 接口，
 * 替代各管线中分散的 OpenCV CPU 预处理调用。
 *
 * 支持：
 *   1. BGR→RGB 颜色空间转换
 *   2. NV12→BGR / NV12→RGB（YUV 到 RGB 硬转换）
 *   3. 任意尺寸 resize（双线性插值）
 *   4. DMA-BUF fd 直通输入（零拷贝，避免 GPU↔CPU 搬运）
 *   5. 回退到 OpenCV CPU 路径（当 RGA 硬件不可用时自动降级）
 *
 * 对应优化方案 阶段B：RGA 全链路接入
 */

// 标准库头文件
#include <cstdint>   // uint8_t / int 等定宽类型
#include <atomic>    // std::atomic<bool> 原子布尔值

// OpenCV 头文件
#include <opencv2/core/mat.hpp>  // cv::Mat 图像矩阵类型

namespace pipeline {  // 管线命名空间

/**
 * @class RgaSubmissionGuard
 * @brief RGA 公平提交锁（RAII 守护）
 *
 * 驱动稳定性要求应用侧串行提交，但普通 std::mutex 不保证公平：高帧率采集线程
 * 会反复抢锁，导致编码和推理等待数秒。该 guard 使用 FIFO ticket 调度，在保持
 * 单任务提交的同时避免线程饥饿。
 *
 * 用法：在需要调用 RGA 的代码块作用域构造 RgaSubmissionGuard，
 *        离开作用域时自动释放。
 */
class RgaSubmissionGuard {
public:
    RgaSubmissionGuard();   // 构造函数：获取 ticket 并等待轮到自己（FIFO 公平调度）
    ~RgaSubmissionGuard();  // 析构函数：释放锁，通知下一个等待者

    // 禁止拷贝（锁语义不可复制）
    RgaSubmissionGuard(const RgaSubmissionGuard&) = delete;
    RgaSubmissionGuard& operator=(const RgaSubmissionGuard&) = delete;

private:
    uint64_t ticket_ = 0;  // 当前持有的 ticket 编号，用于 FIFO 调度
};

/// 检查 RGA 硬件运行时是否健康（驱动是否正常响应）
bool rgaRuntimeHealthy();

/// 报告 RGA 操作结果，用于统计成功/失败比率（监控用）
void reportRgaResult(bool success);

/**
 * @enum RgaPixelFormat
 * @brief RGA 支持的像素格式（子集）
 *
 * 列出 RGA 硬件加速支持的常用像素格式。
 */
enum class RgaPixelFormat : uint8_t {
    BGR888 = 0,    // BGR 8-8-8 格式（OpenCV 默认彩色格式）
    RGB888,        // RGB 8-8-8 格式（深度学习模型常用输入格式）
    NV12,          // YUV 4:2:0 半平面格式（摄像头/编码器标准输出格式）
    NV21,          // YUV 4:2:0 半平面格式（NV12 的 UV 交换版本）
    RGBA8888,      // RGBA 8-8-8-8 带 alpha 通道
    BGRA8888,      // BGRA 8-8-8-8 带 alpha 通道
};

/**
 * @enum RgaColorSpace
 * @brief RGA 颜色空间枚举
 *
 * 指定 YUV↔RGB 转换的矩阵系数，影响颜色还原的准确性。
 */
enum class RgaColorSpace : uint8_t {
    Bt601Limited = 0,  // BT.601 标准，Limited Range (16-235)，标清用
    Bt601Full,         // BT.601 标准，Full Range (0-255)
    Bt709Limited,      // BT.709 标准，Limited Range (16-235)，高清用（推荐默认值）
};

/**
 * @struct RgaPreprocessConfig
 * @brief RGA 预处理配置
 *
 * 封装一次 RGA 预处理操作的所有参数：
 * 源/目标格式、目标尺寸、颜色空间、回退策略。
 */
struct RgaPreprocessConfig {
    bool use_rga = true;                // 是否启用 RGA 硬件（false 时始终回退 CPU）
    bool strict_hardware = false;       // 严格硬件模式：RGA 不可用/失败时不回退 CPU，直接返回失败
    int  target_width  = 640;           // 模型输入宽度（像素）
    int  target_height = 640;           // 模型输入高度（像素）
    RgaPixelFormat src_format = RgaPixelFormat::BGR888;  // 源图像像素格式
    RgaPixelFormat dst_format = RgaPixelFormat::RGB888;  // 目标图像像素格式（模型期望格式）
    RgaColorSpace color_space = RgaColorSpace::Bt709Limited; // YUV↔RGB 颜色空间转换矩阵
};

/**
 * @class RgaPreprocessor
 * @brief RGA 硬件加速预处理器
 *
 * 提供统一的图像预处理接口，内部自动选择 RGA 硬件路径或 CPU 回退路径。
 *
 * 生命周期：创建后 initialize()，后续调用 process() 即可。
 * 线程安全：同一实例不可在多线程并发调用 process()，
 *           每条管线应持有自己的实例。
 */
class RgaPreprocessor {
public:
    RgaPreprocessor();   // 默认构造函数
    ~RgaPreprocessor();  // 析构函数：释放 RGA 上下文资源

    // non-copyable（管理硬件资源，不可复制）
    RgaPreprocessor(const RgaPreprocessor&) = delete;
    RgaPreprocessor& operator=(const RgaPreprocessor&) = delete;

    /**
     * @brief 初始化预处理器
     * @param config 预处理配置
     * @return true 成功（RGA 可用或 CPU 回退就绪）；false 严格模式下 RGA 不可用
     */
    bool initialize(const RgaPreprocessConfig& config);

    /**
     * @brief 从 cv::Mat 输入进行预处理（resize + cvtColor）
     * @param src 输入图像（BGR/NV12 格式，由 src_format 指定）
     * @param dst 输出图像（RGB/BGR 格式，由 dst_format 指定，自动分配为 target 尺寸）
     * @return true 成功；false 预处理失败
     */
    bool process(const cv::Mat& src, cv::Mat& dst);

    /**
     * @brief 从 cv::Mat 输入直接写入外部目标缓冲区
     * @param src              输入图像（BGR/NV12）
     * @param dst_data         外部目标缓冲首地址（调用方预分配）
     * @param dst_stride       目标缓冲行步长（字节）
     * @param dst_height_stride 目标垂直步长；0 表示可见高度
     * @return true 成功
     *
     * 主要用于输出到编码器/外部复用缓冲，避免额外的中间 Mat 再拷贝。
     */
    bool processToBuffer(const cv::Mat& src, uint8_t* dst_data, int dst_stride,
                         int dst_height_stride = 0);

    /**
     * @brief 从 DMA-BUF fd 输入进行预处理（零拷贝路径）
     * @param src_fd          输入 DMA-BUF 文件描述符
     * @param src_width       输入帧宽度（像素）
     * @param src_height      输入帧高度（像素）
     * @param src_stride      输入行步长（字节）
     * @param dst             输出 cv::Mat（自动分配为 target 尺寸）
     * @param src_height_stride 输入垂直步长；0 表示可见高度
     * @param src_buffer_size   DMA-BUF 缓冲区总大小（字节）
     * @return true 成功
     *
     * 不经过 CPU 拷贝像素，直接从 DMA-BUF fd 导入 RGA 硬件处理，
     * 是最高性能的预处理路径。
     */
    bool processFromFd(int src_fd, int src_width, int src_height, int src_stride,
                       cv::Mat& dst, int src_height_stride = 0,
                       size_t src_buffer_size = 0);

    /// BGR 图像硬件旋转
    /// @param src 输入 BGR 图像
    /// @param dst 输出旋转后的 BGR 图像
    /// @param rotation 顺时针旋转角度：0/90/180/270
    /// @return true 成功
    bool rotateBgr(const cv::Mat& src, cv::Mat& dst, int rotation) const;

    /// 是否实际使用了 RGA（还是回退了 CPU 路径）
    bool isRgaActive() const { return rga_available_; }

private:
    /// OpenCV CPU 回退实现（resize + cvtColor）
    bool processCpu(const cv::Mat& src, cv::Mat& dst);

    /// RGA 硬件实现（resize + cvtColor，通过 im2d API）
    bool processRga(const cv::Mat& src, cv::Mat& dst);

    /// RGA 硬件实现：直接写入外部目标缓冲（避免中间 Mat 拷贝）
    bool processRgaToBuffer(const cv::Mat& src, uint8_t* dst_data, int dst_stride,
                            int dst_height_stride);

    /// RGA DMA-BUF fd 输入实现（零拷贝路径的核心实现）
    bool processRgaFd(int src_fd, int src_width, int src_height, int src_stride,
                      cv::Mat& dst, int src_height_stride, size_t src_buffer_size);

    RgaPreprocessConfig config_;      // 预处理配置副本
    bool initialized_   = false;      // 是否已完成初始化
    bool rga_available_  = false;     // RGA 硬件是否可用（探测结果）
};

} // namespace pipeline
