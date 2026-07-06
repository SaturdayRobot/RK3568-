/**
 * @file mpp_encoder.h
 * @brief MPP H.264 硬件编码器类定义
 *
 * 该文件定义了 MppH264Encoder 类，封装 Rockchip MPP 的 H.264 硬件编码功能。
 *
 * 主要特性：
 * - 使用 DRM（Direct Rendering Manager）缓冲区实现零拷贝输入
 * - 支持 CBR（恒定码率）速率控制
 * - 自动生成并缓存 SPS/PPS 头信息（H.264 Annex B 格式）
 * - 16 字节对齐的宽高步长，满足 VEPU 硬件编码器的对齐要求
 * - 通过 metadata 检测硬件实际产出的帧类型（I帧/P帧）
 *
 * 编码管道：
 * 外部写入NV12 -> MppFrame封装 -> encode_put_frame() -> VEPU硬件编码器
 *    -> encode_get_packet() -> H.264 Annex B码流(含SPS/PPS头)
 */

#pragma once

#include <cstdint>   // 标准整数类型定义
#include <vector>    // 用于存储编码后的码流头和码流数据

#include "rk_mpi.h"  // Rockchip MPP 主接口头文件，包含 MppCtx/MppApi/MppEncCfg 等核心类型

/**
 * @class MppH264Encoder
 * @brief 基于 Rockchip MPP 的 H.264 硬件编码器封装
 *
 * 该类封装了 Rockchip MPP（Media Process Platform）的 H.264 硬件编码功能，
 * 底层使用 RK3568 的 VEPU（Video Encoder Processing Unit）硬件模块。
 *
 * 主要设计特点：
 * - 帧缓冲区复用：整个生命周期仅分配一次输入缓冲区，避免每帧 malloc/free
 * - 对齐填充预初始化：Init 时一次性填入灰底色，解决 RGA 局部写入导致的编码伪影
 * - 零拷贝管线：RGA(dma_fd) -> VENC(dma_fd) 全程无 CPU 拷贝
 * - SPS/PPS 缓存：首次初始化后缓存在 header_ 中，后续关键帧前直接插入
 *
 * 典型使用流程：
 * 1. 调用 initialize() 初始化和配置编码器
 * 2. 通过 inputData()/inputFd() 获取输入缓冲区地址/fd，由外部（如RGA）填充YUV数据
 * 3. 调用 encode() 执行一帧编码
 * 4. 析构时自动调用 shutdown() 释放所有 MPP 资源
 *
 * @note 禁止拷贝（= delete），因为 MPP 内部句柄不可共享
 */
class MppH264Encoder {
public:
    MppH264Encoder() = default;                     // 默认构造函数，所有成员通过 initialize() 初始化
    ~MppH264Encoder();                              // 析构函数，调用 shutdown() 释放所有 MPP 资源
    MppH264Encoder(const MppH264Encoder&) = delete; // 禁止拷贝构造，MPP 上下文/缓冲区组不能跨对象共享
    MppH264Encoder& operator=(const MppH264Encoder&) = delete; // 禁止拷贝赋值

    /**
     * @brief 初始化 H.264 硬件编码器
     * @param width  视频宽度（像素），无需对齐，内部自动 align16
     * @param height 视频高度（像素），无需对齐
     * @param fps    目标帧率，同时决定 GOP 大小（GOP = fps，即每秒一个 IDR）
     * @param bitrate 目标比特率（bps），用于 CBR 速率控制
     * @return true 初始化成功，false 初始化失败（SPS/PPS 获取失败或资源分配失败）
     *
     * 初始化步骤（共4步）：
     * 1. 创建 DRM 缓冲区组，分配输入帧缓冲和码流包缓冲
     * 2. 创建 MPP 编码器上下文并初始化为 H.264 编码类型
     * 3. 配置编码参数：分辨率/格式/CBR/GOP/QP/H.264 profile/level/CABAC 等
     * 4. 获取并缓存 SPS/PPS 头信息
     * 5. 预填充输入缓冲区的对齐填充区域（灰底色）
     */
    bool initialize(int width, int height, int fps, int bitrate);

    /**
     * @brief 关闭编码器并释放所有资源
     *
     * 释放顺序（严格）：配置句柄 -> MPP 上下文 -> 帧缓冲区 -> 码流包缓冲区 -> 缓冲区组
     * 顺序不可颠倒：先释放 MPP 上下文再释放缓冲区会导致 MPP 内部断言失败。
     */
    void shutdown();

    /**
     * @brief 获取输入帧缓冲区的虚拟地址指针，供外部模块（如 RGA）写入 YUV 数据
     * @return 指向 NV12（YUV420SP）帧缓冲区的内核映射虚拟地址，未初始化时返回 nullptr
     *
     * @note 该指针在编码器生命周期内有效。写入应在 encode() 调用前完成，
     *       且不应与 encode() 并发执行（同一帧的写入和编码须串行化）。
     */
    uint8_t* inputData() const;

    /**
     * @brief 获取输入帧缓冲区的 DMA 文件描述符，用于零拷贝传递
     * @return DMA buffer fd，未初始化或分配失败时返回 -1
     *
     * 典型用法：将 fd 传递给 RGA 的 src/dst import，
     * RGA 通过 DRM_PRIME 机制直接操作同一块物理内存，无需 CPU 拷贝。
     */
    int inputFd() const;

    int stride() const { return stride_; }               // 返回水平步长（=align16(width)），单位：字节
    int verticalStride() const { return vertical_stride_; } // 返回垂直步长（=align16(height)），单位：像素行

    /**
     * @brief 获取缓存的 SPS/PPS 头信息（H.264 Annex B 格式）
     * @return SPS+PPS NAL 单元序列的常量引用
     *
     * 每次编码 IDR 关键帧前，应先将此头信息插入码流，
     * 使解码器能在任意 IDR 帧处初始化解码。
     */
    const std::vector<uint8_t>& header() const { return header_; }

    /**
     * @brief 编码一帧 YUV 数据为 H.264 码流
     * @param[in]  pts      显示时间戳（Presentation Time Stamp）
     *                      1. 用于判断 IDR 帧：pts % gop == 0 时预判为关键帧
     *                      2. 编码后写入码流，解码端用于音视频同步
     * @param[out] packet   编码后的 H.264 Annex B 码流数据（不含 SPS/PPS 头，
     *                      调用方应在前级自行插入 header()）
     * @param[out] key_frame 输出参数：
     *                      输入时：基于 pts%gop 预判的值
     *                      输出时：通过 metadata 读取的硬件实际帧类型（修正值）
     * @return true 编码成功并产出码流，false 编码失败或未初始化
     *
     * 编码流程（4步）：
     * 1. 根据 PTS 和 GOP 预判是否为关键帧
     * 2. 创建 MppFrame，设置宽高/步长/格式/PTS，关联复用缓冲区
     * 3. encode_put_frame 投喂 -> encode_get_packet 获取输出
     * 4. 通过 metadata 的 KEY_OUTPUT_INTRA 读取硬件实际帧类型，修正 key_frame
     *
     * @note 调用本函数前，外部必须已通过 inputData()/inputFd() 写入新的 YUV 数据
     */
    bool encode(int64_t pts, std::vector<uint8_t>& packet, bool& key_frame);

private:
    MppCtx ctx_ = nullptr;              // MPP 编码器上下文句柄，所有 MPP 操作的核心对象
    MppApi* mpi_ = nullptr;             // MPP API 函数表指针，提供 control/encode_put_frame/encode_get_packet
    MppEncCfg cfg_ = nullptr;           // 编码器配置句柄，用于读写编码参数（分辨率/码率/GOP/QP/profile等）
    MppBufferGroup group_ = nullptr;    // MPP 缓冲区组，管理一组 DRM 类型缓冲区
    MppBuffer frame_buffer_ = nullptr;  // 输入帧缓冲区（YUV420SP/NV12 格式），全生命周期复用
    MppBuffer packet_buffer_ = nullptr; // 码流包缓冲区，仅用于 SPS/PPS 头信息获取阶段
    int width_ = 0;                     // 视频宽度（像素），用户指定值
    int height_ = 0;                    // 视频高度（像素），用户指定值
    int stride_ = 0;                    // 水平步长 = align16(width)，用于 VEPU 硬件对齐
    int vertical_stride_ = 0;           // 垂直步长 = align16(height)，用于 VEPU 硬件对齐
    int fps_ = 0;                       // 帧率，同时用作 GOP 大小（GOP=fps，即每秒一个关键帧）
    int gop_ = 0;                       // 图像组大小（GOP），两个 IDR 帧之间的帧间隔
    std::vector<uint8_t> header_;       // 缓存的 SPS+PPS 头信息（H.264 Annex B 格式），在关键帧前插入码流
};
