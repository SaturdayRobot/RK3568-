/**
 * @file mpp_encoder.cpp
 * @brief MPP H.264 硬件编码器实现
 *
 * 该文件实现 MppH264Encoder 类，封装 Rockchip MPP 硬件编码器的完整生命周期：
 * 初始化 -> 参数配置 -> 获取 SPS/PPS -> 逐帧编码 -> 资源释放。
 *
 * 编码管道：
 * YUV420SP(NV12) 输入缓冲区 -> encode_put_frame() -> MPP 硬件编码器 -> encode_get_packet() -> H.264 码流
 *
 * 缓冲区策略：
 * - 复用同一块 DRM 帧缓冲区，每帧只更新可见区域，减少分配开销
 * - 对齐填充区域在初始化时一次性写入灰底色，避免底部脏边（绿色/紫色伪影）
 * - encode_put_frame 投喂后立即 deinit MppFrame，但底层 MppBuffer 仍被编码器异步持有
 */

#include "data_processing/mpp_encoder.h"

#include <algorithm>  // std::max
#include <cstring>    // std::memset
#include <iostream>   // std::cerr, std::cout

#include "mpp_buffer.h"   // MppBuffer/MppBufferGroup 操作：mpp_buffer_get_ptr/mpp_buffer_get_fd 等
#include "rk_venc_cfg.h"  // 编码器配置键定义：prep:width, rc:mode, h264:profile 等
#include "mpp_frame.h"    // MppFrame 操作：mpp_frame_init/mpp_frame_set_* 系列 API
#include "mpp_packet.h"   // MppPacket 操作：mpp_packet_get_pos/mpp_packet_get_length 等

namespace {
/**
 * @brief 将值向上对齐到 16 字节边界
 * @param value 原始值
 * @return 对齐后的值（value + 15 后清除低 4 位）
 *
 * RK3568 MPP 硬件编码器（VEPU）要求水平/垂直步长为 16 的整数倍，
 * 不对齐会导致编码失败或图像花屏。这是 MPP 驱动层的硬性约束。
 */
int align16(int value) { return (value + 15) & ~15; }
}  // namespace

/**
 * @brief 析构函数，确保编码器资源被正确释放
 */
MppH264Encoder::~MppH264Encoder() { shutdown(); }

bool MppH264Encoder::initialize(int width, int height, int fps, int bitrate) {
    // 如果之前已初始化，先释放旧资源避免泄漏
    shutdown();

    // 保存用户指定的视频参数
    width_ = width;                         // 视频宽度（像素）
    height_ = height;                       // 视频高度（像素）
    stride_ = align16(width);               // 水平步长：16字节对齐（Y平面每个像素1字节）
    vertical_stride_ = align16(height);     // 垂直步长：16字节对齐
    fps_ = std::max(1, fps);                // 帧率最小为1，防止除零
    gop_ = fps_;                            // GOP = 帧率，即每秒一个IDR关键帧

    // 计算 NV12 格式所需的总帧大小：Y平面（stride*ver_stride）+ UV交错平面（stride*ver_stride/2）
    const size_t frame_size = static_cast<size_t>(stride_) * vertical_stride_ * 3 / 2;

    // === 步骤1：创建 MPP 缓冲区组和分配缓冲区 ===
    // MPP_BUFFER_TYPE_DRM：使用 DRM（Direct Rendering Manager）分配器
    // 优势：支持 dma-buf (DMA Buffer) 导出，可在不同硬件模块间（RGA->VENC）零拷贝传递
    // 调用链：mpp_buffer_group_get_internal -> 创建缓冲区组 -> mpp_buffer_get -> 从组内分配单块缓冲区
    //
    // 注意：这里将多个 MPP 操作串联在 if 条件中，任一失败即停止后续操作，
    // 通过短路求值避免在无效对象上继续调用。
    if (mpp_buffer_group_get_internal(&group_, MPP_BUFFER_TYPE_DRM) != MPP_OK ||  // 创建DRM类型缓冲区组
        mpp_buffer_get(group_, &frame_buffer_, frame_size) != MPP_OK ||           // 从组内分配输入帧缓冲区
        mpp_buffer_get(group_, &packet_buffer_, frame_size) != MPP_OK ||          // 分配码流包缓冲区（用于SPS/PPS获取）
        mpp_create(&ctx_, &mpi_) != MPP_OK || !ctx_ || !mpi_ ||                   // 创建MPP编码器上下文和API接口实例
        mpp_init(ctx_, MPP_CTX_ENC, MPP_VIDEO_CodingAVC) != MPP_OK ||             // 初始化上下文为H.264编码器类型
        mpp_enc_cfg_init(&cfg_) != MPP_OK ||                                      // 创建编码器配置句柄
        mpi_->control(ctx_, MPP_ENC_GET_CFG, cfg_) != MPP_OK) {                   // 从编码器获取默认配置模板
        std::cerr << "[MppEncoder] resource initialization failed\n";
        shutdown();
        return false;
    }

    // === 步骤2：配置编码器参数 ===
    // 每个 mpp_enc_cfg_set_s32 调用设置一个编码参数键值对
    // 参数键格式： "模块:参数名"，如 "prep:width" 表示预处理模块的宽度参数
    // 这些参数将影响 VEPU（Video Encoder Processing Unit）硬件的寄存器配置

    // --- 预处理（prep）参数：输入图像格式和尺寸 ---
    // prep 模块负责将应用层输入的 NV12 帧数据对齐后送入编码核
    mpp_enc_cfg_set_s32(cfg_, "prep:width", width_);               // 输入图像宽度
    mpp_enc_cfg_set_s32(cfg_, "prep:height", height_);             // 输入图像高度
    mpp_enc_cfg_set_s32(cfg_, "prep:hor_stride", stride_);         // 水平步长（Y平面每行字节数）
    mpp_enc_cfg_set_s32(cfg_, "prep:ver_stride", vertical_stride_); // 垂直步长
    mpp_enc_cfg_set_s32(cfg_, "prep:format", MPP_FMT_YUV420SP);    // 像素格式：YUV420SP = NV12（Y平面后接交错UV平面）

    // --- 速率控制（rc）参数：CBR恒定位速率模式 ---
    // 速率控制模块决定了 QP 值的动态调整策略
    mpp_enc_cfg_set_s32(cfg_, "rc:mode", MPP_ENC_RC_MODE_CBR);     // 速率控制模式：CBR（Constant Bit Rate），保持码率恒定
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_flex", 0);                // 输入帧率固定模式（0=固定，不动态调整）
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_num", fps_);              // 输入帧率分子
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_denorm", 1);              // 输入帧率分母（fps = num/denorm）
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_flex", 0);               // 输出帧率固定模式
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_num", fps_);             // 输出帧率分子
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_denorm", 1);             // 输出帧率分母
    mpp_enc_cfg_set_s32(cfg_, "rc:bps_target", bitrate);           // 目标比特率（bps），速率控制器将围绕此值调整
    mpp_enc_cfg_set_s32(cfg_, "rc:bps_max", bitrate * 17 / 16);    // 最大比特率：目标 * 17/16，允许小幅波动
    mpp_enc_cfg_set_s32(cfg_, "rc:bps_min", bitrate * 15 / 16);    // 最小比特率：目标 * 15/16
    mpp_enc_cfg_set_s32(cfg_, "rc:gop", gop_);                     // GOP大小（IDR关键帧间隔），影响随机访问延迟
    mpp_enc_cfg_set_s32(cfg_, "rc:qp_init", -1);                   // 初始QP值：-1=让编码器根据码率和分辨率自动决定
    mpp_enc_cfg_set_s32(cfg_, "rc:qp_max", 36);                    // 最大QP值（质量下限，值越大质量越低、码率越小）
    mpp_enc_cfg_set_s32(cfg_, "rc:qp_min", 10);                    // 最小QP值（质量上限，值越小质量越高、码率越大）
    mpp_enc_cfg_set_s32(cfg_, "rc:qp_max_i", 36);                  // I帧最大QP值
    mpp_enc_cfg_set_s32(cfg_, "rc:qp_min_i", 10);                  // I帧最小QP值
    mpp_enc_cfg_set_s32(cfg_, "rc:qp_ip", 2);                      // I帧和P帧之间的QP偏移量

    // --- 编码器类型和 H.264 特定参数 ---
    mpp_enc_cfg_set_s32(cfg_, "codec:type", MPP_VIDEO_CodingAVC);  // 编码类型：AVC = H.264
    mpp_enc_cfg_set_s32(cfg_, "h264:profile", 100);                // H.264 Profile：100 = High Profile（支持8x8变换等高级特性）
    mpp_enc_cfg_set_s32(cfg_, "h264:level", 40);                   // H.264 Level：40 = Level 4.0（支持 1080p@30fps 及以下）
    mpp_enc_cfg_set_s32(cfg_, "h264:cabac_en", 1);                 // 启用 CABAC 熵编码（比 CAVLC 压缩率高约10%，但解码复杂度更高）

    // 将配置写回编码器上下文，使其生效
    // MPP_ENC_SET_CFG：将配置键值对批量写入硬件寄存器，之后输入的帧都按此配置编码
    if (mpi_->control(ctx_, MPP_ENC_SET_CFG, cfg_) != MPP_OK) {
        std::cerr << "[MppEncoder] MPP_ENC_SET_CFG failed\n";
        shutdown();
        return false;
    }

    // 设置头信息模式：每个 IDR 帧前插入 SPS/PPS
    // MPP_ENC_HEADER_MODE_EACH_IDR：编码器在每个 IDR 帧前自动输出 SPS/PPS NAL 单元，
    // 确保解码器可在任意 IDR 帧处开始解码（适合点播/直播场景）
    MppEncHeaderMode header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
    mpi_->control(ctx_, MPP_ENC_SET_HEADER_MODE, &header_mode);

    // === 步骤3：获取 SPS/PPS 头信息 ===
    // SPS（Sequence Parameter Set）和 PPS（Picture Parameter Set）是 H.264 解码必需的参数集
    // - SPS 包含：分辨率、Profile/Level、帧率、参考帧数等序列级参数
    // - PPS 包含：熵编码模式（CABAC/CAVLC）、量化参数、分片模式等图像级参数
    // 必须在关键帧前发送，否则解码器无法正确解码
    MppPacket hdr = nullptr;
    if (mpp_packet_init_with_buffer(&hdr, packet_buffer_) != MPP_OK) {  // 复用预分配的packet_buffer，避免额外分配
        shutdown();
        return false;
    }
    mpp_packet_set_length(hdr, 0);  // 重置包长度，让编码器从头填充
    // MPP_ENC_GET_HDR_SYNC：同步获取 SPS/PPS 头信息（阻塞等待编码器生成）
    // 注意：此调用必须在 MPP_ENC_SET_CFG 之后、首次 encode_put_frame 之前执行
    const MPP_RET hdr_ret = mpi_->control(ctx_, MPP_ENC_GET_HDR_SYNC, hdr);
    if (hdr_ret == MPP_OK) {
        // 从码流包中提取 SPS/PPS 字节数据并缓存
        auto* ptr = static_cast<uint8_t*>(mpp_packet_get_pos(hdr));    // 获取码流数据起始位置
        const size_t len = mpp_packet_get_length(hdr);                  // 获取码流数据长度
        header_.assign(ptr, ptr + len);                                 // 复制到成员变量缓存
    }
    mpp_packet_deinit(&hdr);  // 释放临时头信息包（底层 packet_buffer_ 的引用计数-1，不会释放缓冲区）
    if (header_.empty()) {
        std::cerr << "[MppEncoder] failed to obtain SPS/PPS\n";
        shutdown();
        return false;
    }

    // === 步骤4：固定初始化对齐填充区 ===
    // RGA每帧只更新可见区域，编码器仍会按ver_stride定位UV平面。
    // 底部对齐填充区域如果残存随机值，编码后可能出现绿色/紫色脏边。
    // 解决方案：用灰底色（Y=16 黑电平, UV=128 无色差）填充整个缓冲区。
    // 原理：H.264 编码器使用整帧（含对齐填充行）进行运动搜索/补偿，
    // 如果填充区域是随机数据，会被当成"运动区域"导致压缩效率下降和编码伪影。
    if (auto* input = inputData()) {
        mpp_buffer_sync_begin(frame_buffer_);
        const size_t y_size = static_cast<size_t>(stride_) * vertical_stride_;  // Y平面大小
        std::memset(input, 16, y_size);              // Y平面填充为16（黑电平），避免亮度噪点
        std::memset(input + y_size, 128, y_size / 2); // UV平面填充为128（无色差），避免彩色伪影
        mpp_buffer_sync_end(frame_buffer_);
    }

    std::cout << "[MppEncoder] initialized H.264 " << width_ << 'x' << height_
              << '@' << fps_ << " input_fd=" << inputFd() << '\n';
    return true;
}

/**
 * @brief 释放编码器所有 MPP 资源
 *
 * 释放顺序必须严格遵循：先 deinit 配置，再 destroy 上下文，最后释放缓冲区。
 * 原因：MPP 上下文销毁时，内部仍可能持有对缓冲区的引用（如最后的输出帧缓冲），
 * 如果先释放缓冲区再销毁上下文，MPP 内部会检测到非法引用并触发断言失败。
 */
void MppH264Encoder::shutdown() {
    header_.clear();                               // 清空缓存的SPS/PPS头
    if (cfg_) { mpp_enc_cfg_deinit(cfg_); cfg_ = nullptr; }     // 释放编码配置句柄
    if (ctx_) { mpp_destroy(ctx_); ctx_ = nullptr; mpi_ = nullptr; } // 销毁MPP上下文（mpi_随之失效）
    if (frame_buffer_) { mpp_buffer_put(frame_buffer_); frame_buffer_ = nullptr; }   // 归还输入帧缓冲区（引用计数-1）
    if (packet_buffer_) { mpp_buffer_put(packet_buffer_); packet_buffer_ = nullptr; } // 归还码流包缓冲区
    if (group_) { mpp_buffer_group_put(group_); group_ = nullptr; }                   // 销毁缓冲区组（同时释放组内所有缓冲区）
}

/**
 * @brief 获取输入帧缓冲区的虚拟地址指针
 *
 * mpp_buffer_get_ptr：获取 MppBuffer 对应的内核映射虚拟地址。
 * 外部模块（如 RGA）可通过该指针直接写入 YUV 数据，无需额外拷贝。
 * 注意：该指针在整个编码器生命周期内有效（缓冲区不释放），
 * 但写入操作应与 encode() 调用串行化（同一帧的写入和编码不可并发）。
 */
uint8_t* MppH264Encoder::inputData() const {
    return frame_buffer_ ? static_cast<uint8_t*>(mpp_buffer_get_ptr(frame_buffer_)) : nullptr;
}

/**
 * @brief 获取输入帧缓冲区的 DMA 文件描述符
 *
 * mpp_buffer_get_fd：导出 DRM 缓冲区的 dma-buf 文件描述符。
 * 外部模块（如 RGA、DRM 显示）可通过该 fd 进行零拷贝硬件加速操作。
 * 使用 DRM_PRIME 机制，fd 可在进程间传递，实现跨进程零拷贝。
 * 典型管线：RGA 通过 fd import 操作写入 NV12 → VEPU 从同一物理内存读取编码。
 */
int MppH264Encoder::inputFd() const {
    return frame_buffer_ ? mpp_buffer_get_fd(frame_buffer_) : -1;
}

size_t MppH264Encoder::inputSize() const {
    return frame_buffer_ ? mpp_buffer_get_size(frame_buffer_) : 0;
}

bool MppH264Encoder::beginInputCpuAccess() {
    return frame_buffer_ && mpp_buffer_sync_begin(frame_buffer_) == MPP_OK;
}

bool MppH264Encoder::endInputCpuAccess() {
    return frame_buffer_ && mpp_buffer_sync_end(frame_buffer_) == MPP_OK;
}

/**
 * @brief 编码一帧 YUV 数据为 H.264 码流
 *
 * 编码管道：
 * 输入NV12缓冲区 -> MppFrame封装 -> encode_put_frame()投喂 -> 硬件编码器 -> encode_get_packet()取回 -> H.264码流
 *
 * 注意：本实现中的 encode_put_frame 和 encode_get_packet 是 1:1 同步的——
 * 投喂一帧后立即等待获取输出。这是因为 MppFrame 在投喂后即 deinit，
 * 而编码器异步持有的是底层 MppBuffer 的引用，不影响 buffer 复用。
 *
 * @param[in]  pts      显示时间戳，用于：
 *                      1. 关键帧判断：pts % gop == 0 时强制IDR帧
 *                      2. 码流中的时间戳，解码端用于音视频同步
 * @param[out] output   输出的 H.264 Annex B 码流（NAL单元序列）
 * @param[out] key_frame 输出参数：true=IDR关键帧，false=P/B帧
 * @return true 编码成功，false 失败
 */
bool MppH264Encoder::encode(int64_t pts, std::vector<uint8_t>& output, bool& key_frame) {
    output.clear();  // 清空输出缓冲区

    // 关键帧判断：当 PTS 能被 GOP 整除时，编码为 IDR 帧
    // 例如 GOP=30, PTS=0,30,60,... 为关键帧，保证每秒一个关键帧
    // 注意：这是应用层的预判，最终由 VEPU 硬件编码器决定帧类型，
    // 在步骤4中会通过 metadata 读取硬件实际产出的帧类型进行修正
    key_frame = (pts % gop_) == 0;

    // 快速失败检查：编码器未初始化或无帧缓冲区时直接返回
    if (!ctx_ || !mpi_ || !frame_buffer_) return false;

    // === 步骤1：创建 MppFrame 并设置属性 ===
    // MppFrame 是 MPP 中帧数据的容器，封装了图像参数和缓冲区引用
    MppFrame frame = nullptr;
    if (mpp_frame_init(&frame) != MPP_OK) return false;  // 分配 MppFrame 对象

    mpp_frame_set_width(frame, width_);                  // 设置帧的显示宽度（像素）
    mpp_frame_set_height(frame, height_);                // 设置帧的显示高度（像素）
    mpp_frame_set_hor_stride(frame, stride_);            // 设置水平步长（=16字节对齐后的宽度）
    mpp_frame_set_ver_stride(frame, vertical_stride_);   // 设置垂直步长
    mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);          // 设置像素格式为 NV12
    mpp_frame_set_pts(frame, pts);                       // 设置显示时间戳

    // 关联预分配的 DRM 帧缓冲区到 MppFrame
    // 该缓冲区在整个编码器生命周期中复用，外部（RGA等）应在 encode() 调用前更新内容
    // 注意：mpp_frame_set_buffer 增加 buffer 的引用计数，mpp_frame_deinit 时减少，
    // 底层 MppBuffer 的实际释放在缓冲区组销毁时发生
    mpp_frame_set_buffer(frame, frame_buffer_);

    // === 步骤2：投喂帧到 MPP 编码器 ===
    // encode_put_frame：将 MppFrame 提交到编码器硬件队列，非阻塞
    // 编码器内部异步处理：VEPU 硬件编码完成后结果通过 encode_get_packet 获取
    // 投喂完成后即可释放 MppFrame 对象（底层 MppBuffer 引用计数仍 > 0，不会被释放）
    const MPP_RET put_ret = mpi_->encode_put_frame(ctx_, frame);
    mpp_frame_deinit(&frame);  // 释放 MppFrame 外壳，底层 MppBuffer 仍被编码器持有
    if (put_ret != MPP_OK) return false;

    // === 步骤3：获取编码后的码流包 ===
    // encode_get_packet：从编码器输出队列取出已编码的 H.264 码流包
    // 该调用为同步操作，会阻塞直到有可用输出
    MppPacket packet = nullptr;
    if (mpi_->encode_get_packet(ctx_, &packet) != MPP_OK || !packet) return false;

    // 提取码流数据
    auto* ptr = static_cast<uint8_t*>(mpp_packet_get_pos(packet));  // 获取码流数据起始地址（MPP内部填充）
    const size_t len = mpp_packet_get_length(packet);                // 获取码流数据长度（字节）
    output.assign(ptr, ptr + len);  // 复制码流数据到输出缓冲区

    // === 步骤4：通过 metadata 检测是否为帧内编码帧 ===
    // MPP 在输出 packet 中附加 metadata，记录编码器实际决策结果。
    // 应用层的 pts%gop 预判可能被速率控制器覆盖（场景变化检测等），
    // 因此必须通过 metadata 获取硬件实际产出的帧类型。
    // KEY_OUTPUT_INTRA：标记当前帧是否为帧内编码（I/IDR帧）
    if (mpp_packet_has_meta(packet)) {  // 检查 packet 是否携带 metadata
        RK_S32 intra = 0;
        // 从 metadata 中读取 KEY_OUTPUT_INTRA 键的值
        // 如果读取成功且值非0，则表示该帧确实是帧内编码帧
        if (mpp_meta_get_s32(mpp_packet_get_meta(packet), KEY_OUTPUT_INTRA, &intra) == MPP_OK)
            key_frame = intra != 0;  // 使用硬件返回的实际值覆盖软件层预判
    }
    mpp_packet_deinit(&packet);  // 释放码流包对象（输出数据已拷贝到 output）

    return !output.empty();  // 有输出数据即认为编码成功
}

bool MppH264Encoder::requestIdr() {
    if (!ctx_ || !mpi_) return false;
    return mpi_->control(ctx_, MPP_ENC_SET_IDR_FRAME, nullptr) == MPP_OK;
}
