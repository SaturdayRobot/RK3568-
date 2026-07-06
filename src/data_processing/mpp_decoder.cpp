/**
 * @file mpp_decoder.cpp
 * @brief MPP 硬件解码器实现
 *
 * 该文件实现 MppDecoder 类，封装 Rockchip MPP（Media Process Platform）
 * 硬件解码器的完整生命周期管理。
 *
 * 解码管道：
 * H.264/H.265 码流 -> decode_put_packet() [投喂] -> MPP 硬件解码器
 *    -> decode_get_frame() [拉取] -> info-change 处理 / 帧回调 -> 应用层
 *
 * 支持的编码格式：
 * - MPP_VIDEO_CodingAVC  (H.264)
 * - MPP_VIDEO_CodingHEVC (H.265)
 * - MPP_VIDEO_CodingMJPEG (MJPEG)
 *
 * 缓冲区策略：
 * - 使用 DRM 类型缓冲区组，支持 dma-buf 导出
 * - info-change 时动态重新配置缓冲区组大小和数量
 * - 通过 shared_ptr token 机制让应用层安全跨线程持有帧数据
 */

#include <stdio.h>     // printf
#include <sched.h>     // sched_yield（让出 CPU 时间片，用于超时重试）
#include <sys/time.h>  // 时间操作（预留）
#include <unistd.h>    // usleep（微秒级睡眠，用于超时重试）
#include <iostream>    // std::cerr（错误日志输出）
#include "data_processing/mpp_decoder.h"

/**
 * @brief 构造函数，仅设置默认值，实际初始化由 Init() 完成
 */
MppDecoder::MppDecoder() {}

/**
 * @brief 析构函数，按安全顺序释放所有 MPP 资源
 *
 * 释放顺序说明（不可颠倒，否则 MPP 内部会产生断言/崩溃）：
 * 1. MppPacket（预创建的数据包对象）
 * 2. MppFrame（解码输出的帧对象）
 * 3. MppCtx（MPP 解码器上下文，必须先于缓冲区组释放——缓冲区组在上下文销毁后引用 ctx 会导致非法访问）
 * 4. MppBufferGroup（DRM 帧缓冲区组，在上下文销毁后释放）
 */
MppDecoder::~MppDecoder()
{
    // 释放预创建的 MppPacket 对象
    if (packet)
    {
        mpp_packet_deinit(&packet);  // 销毁 packet 并置 NULL
        packet = NULL;
    }
    // 释放 MppFrame 对象
    if (frame)
    {
        mpp_frame_deinit(&frame);    // 销毁 frame 并置 NULL
        frame = NULL;
    }
    // 销毁 MPP 解码器上下文（释放内部解码器状态和 DPB 参考帧缓存）
    if (mpp_ctx)
    {
        mpp_destroy(mpp_ctx);        // 销毁上下文，对应的 mpi 接口表也失效
        mpp_ctx = NULL;
    }
    // 释放 DRM 帧缓冲区组（减少引用计数，最后一处引用时真正释放）
    if (loop_data.frm_grp)
    {
        mpp_buffer_group_put(loop_data.frm_grp);  // put = 减少引用计数
        loop_data.frm_grp = NULL;
    }
}

int MppDecoder::Init(int video_type, int fps, void *userdata, int id)
{
    // === 防御式清理：避免重复调用 Init() 导致资源泄漏 ===
    if (packet)
    {
        mpp_packet_deinit(&packet);    // 释放旧 packet
        packet = NULL;
    }
    if (frame)
    {
        mpp_frame_deinit(&frame);      // 释放旧 frame
        frame = NULL;
    }
    if (mpp_ctx)
    {
        mpp_destroy(mpp_ctx);          // 销毁旧上下文
        mpp_ctx = NULL;
    }
    if (loop_data.frm_grp)
    {
        mpp_buffer_group_put(loop_data.frm_grp);  // 释放旧缓冲区组
        loop_data.frm_grp = NULL;
    }

    this->id = id;                       // 保存解码器 ID（多路解码时区分）
    MPP_RET ret = MPP_OK;                // MPP 操作返回值（MPP_OK = 0 表示成功）
    this->userdata = userdata;           // 保存用户数据指针（回调时透传）
    this->fps = fps;                     // 保存帧率
    this->last_frame_time_ms = 0;        // 重置上一帧时间戳

    // === 根据 video_type 设置 MPP 编码类型 ===
    if (video_type == 264)
    {
        mpp_type = MPP_VIDEO_CodingAVC;   // H.264/AVC 编码
    }
    else if (video_type == 265)
    {
        mpp_type = MPP_VIDEO_CodingHEVC;  // H.265/HEVC 编码
    }
    else if (video_type == 26)
    {
        mpp_type = MPP_VIDEO_CodingMJPEG; // MJPEG 编码（Motion JPEG）
    }
    else
    {
        printf("unsupport video_type %d", video_type);
        return -1;  // 不支持的编码类型，返回错误
    }

    // === 步骤1：清零解码循环数据 ===
    memset(&loop_data, 0, sizeof(loop_data));  // 将所有字段清零，避免残留值干扰

    // === 步骤2：创建 MPP 解码器上下文和 API 接口 ===
    // mpp_create 是 MPP 框架的核心入口，同时创建上下文对象和填充 API 函数表
    // 内部会根据 RK3568 平台自动选择硬件解码器（RkvDec 或 VDPU）
    MppDecCfg cfg = NULL;                     // 解码器配置句柄
    mpp_mpi = NULL;                           // API 接口指针（由 mpp_create 填充）
    ret = mpp_create(&mpp_ctx, &mpp_mpi);     // 创建上下文和获取 MPI（Media Process Interface）接口表
    if (MPP_OK != ret)
    {
        printf("mpp_create failed ");
        return 0;  // mpp_create 失败，返回0
    }

    // === 步骤3：初始化解码器上下文 ===
    // MPP_CTX_DEC：声明上下文为解码器类型（非编码器）
    // mpp_type：指定要解码的编码格式（AVC/HEVC/MJPEG），MPP 据此选择硬件解码核
    ret = mpp_init(mpp_ctx, MPP_CTX_DEC, mpp_type);
    if (ret)
    {
        printf("%p mpp_init failed ", mpp_ctx);
        return -1;  // mpp_init 失败
    }

    // === 步骤4：获取并修改默认解码配置 ===
    mpp_dec_cfg_init(&cfg);  // 初始化解码配置句柄

    // MPP_DEC_GET_CFG：从解码器上下文读取默认配置（MPP 内部填充的硬件相关默认值）
    ret = mpp_mpi->control(mpp_ctx, MPP_DEC_GET_CFG, cfg);
    if (ret)
    {
        printf("%p failed to get decoder cfg ret %d ", mpp_ctx, ret);
        return -1;
    }

    // 设置 split_parse 参数
    // split_parse = 1：启用 MPP 内部帧分割，自动扫描 NAL 起始码 (0x00000001/0x000001)
    //                   分割输入码流。适用于外部不做 NAL 分割的裸 H.264/H.265 流。
    // split_parse = 0：输入已由外部（如 BSF 模块）做好帧分割，MPP 直接使用。
    //                   当前 need_split = 0，即依赖上层已做好 NAL 单元分割，降低 MPP 内部开销。
    ret = mpp_dec_cfg_set_u32(cfg, "base:split_parse", need_split);
    if (ret)
    {
        printf("%p failed to set split_parse ret %d ", mpp_ctx, ret);
        return -1;
    }

    // MPP_DEC_SET_CFG：将修改后的配置写回解码器上下文，使其立即生效
    ret = mpp_mpi->control(mpp_ctx, MPP_DEC_SET_CFG, cfg);
    if (ret)
    {
        printf("%p failed to set cfg %p ret %d ", mpp_ctx, cfg, ret);
        return -1;
    }

    // === 步骤5：设置输出超时参数 ===
    // MPP_SET_OUTPUT_TIMEOUT：设置 decode_get_frame 的超时时间（毫秒）
    // 当解码器内部无可用帧时，该调用最多等待 output_timeout_ms 毫秒后返回 MPP_ERR_TIMEOUT。
    // 降低此值可减少 get_frame 空转等待带来的 CPU 抖动——超时后线程可立即执行其他任务。
    // 注意：底层硬件解码器可能不支持此参数，不支持时仅打印日志，不影响功能。
    RK_S64 output_timeout_ms = 2;      // 2毫秒超时：平衡响应速度和CPU占用
    ret = mpp_mpi->control(mpp_ctx, MPP_SET_OUTPUT_TIMEOUT, &output_timeout_ms);
    if (ret)
    {
        std::cerr << "MppDecoder::Init set output timeout failed ret=" << ret << std::endl;
    }

    mpp_dec_cfg_deinit(cfg);  // 释放配置句柄（配置已写入上下文，不再需要）

    // === 步骤6：初始化解码循环运行时数据 ===
    loop_data.ctx = mpp_ctx;   // 绑定解码器上下文
    loop_data.mpi = mpp_mpi;   // 绑定 MPI 接口
    loop_data.eos = 0;         // 初始化为非流结束状态
    loop_data.frame = NULL;    // 当前无解码帧

    // === 步骤7：预创建 MppPacket 对象 ===
    // 在 Decode() 中每次只调用 mpp_packet_set_data 重设数据指针，
    // 避免每帧 init/deinit 带来的内存分配开销（init 会调用 malloc，deinit 调用 free）
    ret = mpp_packet_init(&packet, NULL, 0);  // 初始化为空包（无数据）
    if (MPP_OK != ret)
    {
        std::cerr << "MppDecoder::Init mpp_packet_init failed ret=" << ret << std::endl;
        return -1;
    }

    return 1;  // 初始化成功
}

/**
 * @brief 重置解码器
 *
 * 调用 mpp_mpi->reset() 清除解码器内部状态：
 * - 清空参考帧缓存（DPB, Decoded Picture Buffer）
 * - 重置解码器状态机
 * 典型使用场景：seek 操作后需要丢弃所有已缓存的参考帧。
 *
 * 注意：reset 不会释放外部缓冲区组（frm_grp），
 * 仅清除 MPP 内部的 DPB 引用。新帧将继续从现有缓冲区组中分配。
 */
int MppDecoder::Reset()
{
    if (mpp_mpi != NULL && mpp_ctx != NULL)
    {
        mpp_mpi->reset(mpp_ctx);  // 重置解码器上下文的内部状态
    }
    return 0;
}

/**
 * @brief 注册帧回调函数
 */
int MppDecoder::SetCallback(MppDecoderFrameCallback callback)
{
    this->callback = callback;  // 保存回调函数对象（std::function）
    return 0;
}

int MppDecoder::Decode(uint8_t *pkt_data, int pkt_size, int pkt_eos)
{
    // 快速失败：空数据或无有效长度时直接返回，避免无效调用进入 MPP
    if (pkt_data == NULL || pkt_size <= 0)
        return 0;

    // 获取解码循环数据的本地引用
    MpiDecLoopData *data = &loop_data;  // 解码循环运行时数据
    MppCtx ctx = data->ctx;             // MPP 解码器上下文句柄
    MppApi *mpi = data->mpi;            // MPP API 接口函数表
    MPP_RET ret = MPP_OK;               // MPP 操作返回值
    int got_frames = 0;                  // 本轮解码成功获取的帧计数

    // === 复用预创建的 packet：重设数据指针，避免每帧 init/deinit 开销 ===
    // 注意区分三个指针概念：
    //   data:  缓冲区内存基址（MPP 内部用于计算偏移）
    //   pos:   读/写游标位置（MPP 解析时从此处开始读取，解析后自动前移）
    //   length:从 pos 开始的有效数据长度（非从 data 开始）
    // 此处设为三者对齐，表示从起始位置读完整段数据
    mpp_packet_set_data(packet, pkt_data);   // 设置 packet 的内存基址
    mpp_packet_set_size(packet, pkt_size);   // 设置 packet 的缓冲区总容量
    mpp_packet_set_pos(packet, pkt_data);    // 设置 packet 的当前读/写游标位置
    mpp_packet_set_length(packet, pkt_size); // 设置从游标开始的有效数据长度
    if (pkt_eos)
        mpp_packet_set_eos(packet);          // 标记为流结束：解码器收到后将排空内部缓冲帧
    else
        mpp_packet_clr_eos(packet);          // 清除流结束标记

    // === 解码循环的控制常量 ===
    const RK_S32 kMaxPutRetry = 30;       // put_packet 最大重试次数（缓冲区满时）
    const RK_S32 kMaxGetTimeoutRetry = 6; // get_frame 单次调用的超时重试上限
    const RK_S32 kMaxDrainPerRound = 16;  // 每轮最多拉取帧数，防止单次调用长时间占用线程

    RK_S32 put_retry = 0;      // put_packet 当前重试计数
    bool pkt_done = false;     // packet 是否已成功投喂
    bool stream_eos = false;   // 流是否已结束

    while (!stream_eos)  // 主循环：投喂+拉取直到流结束或 packet 处理完成
    {
        // === 阶段1：投喂数据包到 MPP 解码器 ===
        if (!pkt_done)
        {
            // decode_put_packet：将码流数据包提交到解码器硬件队列
            // 非阻塞调用：MPP 接受数据后就返回，实际的硬件解码在后台异步执行
            ret = mpi->decode_put_packet(ctx, packet);
            if (ret == MPP_OK)
            {
                pkt_done = true;  // 投喂成功，标记完成
            }
            else if (ret == MPP_ERR_BUFFER_FULL || ret == MPP_ERR_TIMEOUT)
            {
                // 解码器内部缓冲区已满：等待硬件消费后重试
                if (++put_retry >= kMaxPutRetry)
                {
                    std::cerr << "MPP put_packet retry limit reached (" << kMaxPutRetry << ")" << std::endl;
                    break;  // 达到最大重试次数，放弃本轮解码
                }
                // 自适应等待策略：前3次用 sched_yield 让出 CPU（低延迟，~微秒级）
                // 后续用 usleep(1ms)，避免高频忙等导致的 CPU 飙升
                // 原理：缓冲区满通常意味着硬件解码速度跟不上输入，需要更长的等待时间
                if (put_retry <= 3) {
                    sched_yield();   // 让出 CPU 时间片给其他线程/进程（包括 MPP 硬件中断处理线程）
                } else {
                    usleep(1000);    // 睡眠 1000 微秒（1毫秒），给硬件消费留足时间
                }
                continue;  // 重试投喂
            }
            else
            {
                std::cerr << "decode_put_packet failed ret=" << ret << std::endl;
                break;  // 投喂失败（硬件错误等），退出循环
            }
        }

        // === 阶段2：拉取解码后的帧 ===
        // 一个 packet 可能解码出多帧（如 B 帧重排序场景），需要循环拉取
        RK_S32 drained_count = 0;
        while (drained_count < kMaxDrainPerRound)
        {
            ++drained_count;

            // === 带超时重试的 get_frame ===
            RK_S32 timeout_retry = 0;
            while (1)
            {
                // decode_get_frame：从解码器输出队列获取已解码的帧
                // 此调用根据 Init 中设置的 output_timeout_ms 决定等待时长，
                // 超时未获取到帧时返回 MPP_ERR_TIMEOUT
                ret = mpi->decode_get_frame(ctx, &frame);
                if (ret == MPP_ERR_TIMEOUT && timeout_retry < kMaxGetTimeoutRetry)
                {
                    ++timeout_retry;
                    // 自适应等待：前2次 sched_yield（低延迟），后续 usleep
                    if (timeout_retry <= 2) {
                        sched_yield();
                    } else {
                        usleep(1000);
                    }
                    continue;  // 继续重试
                }
                break;  // 非超时或超过重试上限，退出重试循环
            }

            if (ret == MPP_ERR_TIMEOUT)
            {
                // 当前轮次没有更多可用帧，结束拉取循环
                // 注意：这里跳出的是拉取循环，主循环会继续尝试 put_packet
                break;
            }

            if (ret != MPP_OK)
            {
                // 获取帧失败（硬件错误等），记录日志并结束本轮拉取
                std::cerr << "decode_get_frame failed ret=" << ret << std::endl;
                break;
            }

            if (!frame)
            {
                // MPP_OK 但 frame 为空：当前无更多帧，结束本轮拉取
                break;
            }

            // === 阶段3：处理解码出的帧 ===
            RK_U32 frm_eos = mpp_frame_get_eos(frame);  // 检查帧是否为流结束标志
            bool frame_ok = true;                        // 本帧处理状态标志

            // --- 处理信息变更帧（info-change）---
            // 当码流分辨率等参数发生变化时，MPP 会输出 info-change 帧。
            // 该帧不包含实际图像数据，而是携带新的图像参数（宽、高、缓冲区大小等）。
            // 应用层必须处理 info-change 并回复 MPP_DEC_SET_INFO_CHANGE_READY，
            // 否则 MPP 会一直等待，无法输出后续的正常帧。
            if (mpp_frame_get_info_change(frame))
            {
                // 获取新分辨率所需的缓冲区大小（MPP 根据新分辨率+对齐要求计算）
                RK_U32 buf_size = mpp_frame_get_buf_size(frame);

                // 创建或重新配置 DRM 帧缓冲区组
                if (NULL == data->frm_grp)
                {
                    // 首次 info-change：创建 DRM 类型内部缓冲区组
                    // MPP_BUFFER_TYPE_DRM：使用 DRM（Direct Rendering Manager）分配器
                    // 优势：
                    //   1. 支持 dma-buf 导出（mpp_buffer_get_fd），可在 RGA/DRM/GPU 间零拷贝共享
                    //   2. 物理连续内存分配，满足 VDPU 硬件 DMA 的对齐要求
                    //   3. 内核态统一管理，避免用户态内存映射开销
                    ret = mpp_buffer_group_get_internal(&data->frm_grp, MPP_BUFFER_TYPE_DRM);
                    if (!ret)
                        // MPP_DEC_SET_EXT_BUF_GROUP：将外部缓冲区组注册到解码器
                        // 解码器之后的所有输出帧都将从此缓冲区组中分配内存
                        ret = mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, data->frm_grp);
                }
                else
                {
                    // 非首次 info-change（分辨率再次变更）：
                    // 清空旧缓冲区组，准备重新分配新尺寸的缓冲区
                    ret = mpp_buffer_group_clear(data->frm_grp);
                }

                // 配置缓冲区组限制：每个缓冲区大小 = buf_size，最多 8 个
                // 8 个缓冲区的依据：解码器内部 DPB 最多同时持有约 5~7 帧（含参考帧+当前帧+重排序帧）
                if (!ret)
                    ret = mpp_buffer_group_limit_config(data->frm_grp, buf_size, 8);

                // MPP_DEC_SET_INFO_CHANGE_READY：通知解码器 info-change 已处理完毕
                // 解码器收到此信号后将继续输出正常解码帧
                // 如果应用层不回复此控制命令，解码器将在此处死锁
                if (!ret)
                    ret = mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);

                if (ret)
                {
                    frame_ok = false;
                    std::cerr << "handle info-change failed ret=" << ret << std::endl;
                }
            }
            else
            {
                // --- 处理正常解码帧 ---
                // 获取帧的几何参数
                RK_U32 hor_stride = mpp_frame_get_hor_stride(frame);  // 水平步长：Y平面每行字节数（含对齐填充）
                RK_U32 ver_stride = mpp_frame_get_ver_stride(frame);  // 垂直步长：Y平面总行数（含对齐填充）
                RK_U32 hor_width = mpp_frame_get_width(frame);        // 实际显示宽度（像素，不含对齐填充）
                RK_U32 ver_height = mpp_frame_get_height(frame);      // 实际显示高度（像素，不含对齐填充）

                ++got_frames;  // 成功获取帧计数+1

                // --- 通过回调将帧数据传递给应用程序 ---
                if (callback != nullptr)
                {
                    MppFrameFormat format = mpp_frame_get_fmt(frame);  // 像素格式（如 MPP_FMT_YUV420SP）
                    MppBuffer frame_buf = mpp_frame_get_buffer(frame); // 获取帧的 MppBuffer 句柄
                    // mpp_buffer_get_ptr：获取缓冲区的虚拟地址（内核 DRM 映射到用户空间）
                    // 通过 VA 可直接 CPU 读写，但大分辨率下应优先使用 DMA-fd 避免 CPU 拷贝
                    char *data_vir = frame_buf ? (char *)mpp_buffer_get_ptr(frame_buf) : NULL;
                    // mpp_buffer_get_fd：导出 dma-buf 文件描述符，用于零拷贝跨模块传递
                    // 典型场景：解码器 → RGA（resize/color convert）→ 编码器，全程无 CPU 拷贝
                    int dma_fd = frame_buf ? mpp_buffer_get_fd(frame_buf) : -1;
                    // mpp_buffer_get_size：获取缓冲区总大小（字节）
                    size_t dma_size = frame_buf ? mpp_buffer_get_size(frame_buf) : 0;

                    // --- 帧生命周期管理 ---
                    // 使用 shared_ptr 的自定义删除器实现 RAII 风格的帧生命周期管理：
                    // - 构造 frame_hold_token 时捕获 MppFrame 的引用
                    // - 当所有持有 frame_hold_token 的 shared_ptr 副本析构后，
                    //   自定义删除器自动调用 mpp_frame_deinit 释放 MppFrame 资源
                    // - 这允许应用层跨线程安全地持有帧数据，无需关心释放时机
                    // - 与解码器的内部缓冲区组机制配合：MppFrame 的 deinit 不释放底层
                    //   MppBuffer（缓冲区组持有引用），应用层也可通过 dma_fd 额外持有引用
                    MppFrame frame_hold_ref = frame;
                    std::shared_ptr<void> frame_hold_token(reinterpret_cast<void*>(0x1),
                                                           [frame_hold_ref](void*) mutable {
                                                               if (frame_hold_ref)
                                                               {
                                                                   mpp_frame_deinit(&frame_hold_ref);
                                                               }
                                                           });

                    // 调用回调函数，传递帧的所有必要信息
                    callback(this->userdata,        // 用户数据指针（透传）
                             hor_stride,            // 水平步长
                             ver_stride,            // 垂直步长
                             hor_width,             // 实际宽度
                             ver_height,            // 实际高度
                             format,                // 像素格式
                             dma_fd,                // DMA-BUF fd（-1=不可用）
                             data_vir,              // 虚拟地址（NULL=仅可通过fd访问）
                             dma_size,              // 缓冲区大小
                             this->id,              // 解码器ID
                             frame_hold_token);     // 帧生命周期令牌
                    frame = NULL;  // 所有权已转移给 shared_ptr 删除器，本地指针置空防止 double-free
                }
                else
                {
                    // 无回调注册：直接释放帧
                    mpp_frame_deinit(&frame);
                    frame = NULL;
                }
            }

            if (!frame_ok)
                break;  // 帧处理失败，退出拉取循环

            if (frm_eos)
            {
                // 收到 EOS 帧：流结束，设置标志退出主循环
                stream_eos = true;
                break;
            }
        }  // end of drain loop

        // packet 已成功投喂且已拉取完输出帧，退出主循环
        if (pkt_done)
            break;
    }

    // === 清理 packet 状态，准备下次复用 ===
    // 不释放 packet 本身（预创建的），只重置内部状态
    // 原理：mpp_packet_set_length(0) 让 MPP 认为 packet 已空，配合后续的
    // mpp_packet_set_data 即可安全地重新指向新的输入数据
    mpp_packet_set_length(packet, 0);  // 清零有效数据长度
    mpp_packet_clr_eos(packet);        // 清除 EOS 标志

    return got_frames > 0;  // 返回是否至少解码出一帧
}
