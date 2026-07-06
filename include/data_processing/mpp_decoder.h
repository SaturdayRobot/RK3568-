/**
 * @file mpp_decoder.h
 * @brief MPP 硬件解码器类定义
 *
 * 该文件定义了 MppDecoder 类，封装 Rockchip MPP（Media Process Platform）
 * 硬件视频解码器的完整生命周期管理。
 *
 * 主要功能：
 * 1. 初始化 MPP 解码器（支持 H.264/H.265/MJPEG）
 * 2. 通过回调函数传递解码后的帧数据（支持零拷贝 DMA-BUF）
 * 3. 解码视频数据包（支持分包输入和 NAL 帧分割）
 * 4. 处理解码过程中的 info-change（分辨率动态变更）
 * 5. 重置解码器状态（seek 后清除 DPB）
 *
 * 解码管道：
 * 输入码流包 -> decode_put_packet() -> MPP硬件解码器(VDPU) -> decode_get_frame() -> 回调传递帧数据
 *
 * 零拷贝设计：
 * - DRM 缓冲区组 + dma-buf 导出：解码帧可跨进程/跨硬件模块共享
 * - shared_ptr 生命周期令牌：应用层跨线程安全持有帧数据，无需显式释放
 */

#pragma once
#include <cstring>           // 内存操作
#include "mpp_frame.h"       // MppFrame 类型和操作：mpp_frame_get_buffer/mpp_frame_init 等
#include "rk_mpi.h"          // Rockchip MPP 主接口：MppCtx/MppApi/MppCodingType 等核心类型
#include <functional>        // std::function 用于帧回调函数包装
#include <memory>            // std::shared_ptr 用于跨线程帧生命周期管理
#include <mutex>             // std::mutex 用于线程安全

/**
 * @def FRAME_SYNC_TIME
 * @brief 帧同步等待时间（毫秒），0 表示不启用同步等待
 */
#define FRAME_SYNC_TIME 0

/**
 * @brief MPP 解码器帧回调函数类型定义
 *
 * 解码器每完成一帧解码后，通过此回调将帧数据传递给应用程序。
 * 使用 std::function 支持 lambda、函数指针、bind 表达式等多种回调形式。
 *
 * @param userdata        用户自定义数据指针（在 Init() 中传入，通过回调透传回应用层）
 * @param width_stride    水平步长（字节），即 Y 平面每行的字节数（可能大于实际宽度，含硬件对齐填充）
 * @param height_stride   垂直步长（像素行数），即 Y 平面的总行数（可能大于实际高度）
 * @param width           图像实际显示宽度（像素），不含对齐填充
 * @param height          图像实际显示高度（像素），不含对齐填充
 * @param format          像素格式（MppFrameFormat 枚举值，如 MPP_FMT_YUV420SP = NV12）
 * @param fd              DMA-BUF 文件描述符（-1 表示不可用），用于零拷贝跨模块传递
 * @param data            帧数据的虚拟地址指针（NULL 表示仅可通过 fd 访问）
 * @param buffer_size     缓冲区总大小（字节）
 * @param id              解码器 ID（多路解码时用于区分来源）
 * @param frame_hold_token shared_ptr 生命周期令牌：持有它即可安全跨线程使用帧数据
 *                         当所有持有者释放后，MppFrame 自动 deinit。底层 MppBuffer
 *                         仍由缓冲区组持有引用，直到 Info-Change 或被 decoder 析构。
 */
using MppDecoderFrameCallback = std::function<void(void *userdata,
                                                   int width_stride,
                                                   int height_stride,
                                                   int width,
                                                   int height,
                                                   int format,
                                                   int fd,
                                                   void *data,
                                                   size_t buffer_size,
                                                   int id,
                                                   const std::shared_ptr<void>& frame_hold_token)>;

/**
 * @struct MpiDecLoopData
 * @brief MPP 解码循环数据结构
 *
 * 封装解码过程中需要的所有运行时状态和缓冲区引用。
 * 在 Init() 中初始化，在 Decode() 循环中使用。
 *
 * 关键字段说明：
 * - frm_grp：DRM 帧缓冲区组，info-change 时动态创建/重配置
 * - eos：当 Decode() 收到 pkt_eos=1 时设置，解码器随后排空内部缓冲帧
 */
typedef struct
{
    MppCtx ctx;              ///< MPP 解码器上下文句柄（由 mpp_create 创建，mpp_destroy 释放）
    MppApi *mpi;             ///< MPP API 函数表指针（提供 control/decode_put_packet/decode_get_frame 等接口）
    RK_U32 eos;              ///< 流结束标志（End Of Stream）：1 表示已收到 EOS，解码器将排空 DPB 中的帧
    MppBufferGroup frm_grp;  ///< 帧缓冲区组（DRM 类型），在 info-change 时创建/重新配置
    MppBufferGroup pkt_grp;  ///< 数据包缓冲区组（预留字段，当前未使用）
    MppPacket packet;        ///< 当前待解码的数据包（预留字段，当前通过外部 packet 传入）
    MppFrame frame;          ///< 当前解码出的帧（预留字段，当前通过外部 frame 获取）
    size_t max_usage;        ///< 最大内存使用量跟踪（预留字段）
} MpiDecLoopData;

/**
 * @class MppDecoder
 * @brief MPP 硬件视频解码器封装类
 *
 * 使用 RK3568 MPP 硬件解码器（VDPU，Video Decoder Processing Unit）处理
 * H.264/H.265/MJPEG 视频流。
 *
 * 主要设计特点：
 * - 复用预创建的 MppPacket 对象（每帧只调用 mpp_packet_set_data 重设数据指针）
 * - 支持带超时重试的 put_packet/get_frame 循环，避免忙等导致 CPU 空转
 * - info-change 处理：当分辨率变化时自动重新配置 DRM 缓冲区组
 * - 通过 shared_ptr token 机制实现跨线程安全持有解码帧
 *
 * 典型使用流程：
 * 1. 构造 MppDecoder 对象
 * 2. 调用 Init(video_type, fps, userdata, id) 初始化
 * 3. 调用 SetCallback(cb) 注册帧回调
 * 4. 循环调用 Decode(data, size, eos) 解码视频数据
 * 5. 析构时自动释放所有 MPP 资源
 *
 * @note 析构顺序为 packet -> frame -> ctx -> frm_grp，不可颠倒（MPP 有内部依赖检查）
 */
class MppDecoder
{
public:
    MppCtx mpp_ctx = NULL;   ///< MPP 上下文句柄，公开以便外部在极端情况下直接操作
    MppApi *mpp_mpi = NULL;  ///< MPP API 接口指针，公开以便外部在极端情况下直接调用

    MppDecoder();   ///< 构造函数，仅设置默认值，实际初始化由 Init() 完成
    ~MppDecoder();  ///< 析构函数，依次释放 packet -> frame -> context -> frm_grp

    /**
     * @brief 初始化 MPP 硬件解码器
     * @param video_type 视频编码类型：264=H.264, 265=H.265, 26=MJPEG
     * @param fps        视频帧率（预留，当前未在解码中使用）
     * @param userdata   用户数据指针，通过回调透传给应用层
     * @param id         解码器 ID，多路解码时用于区分来源
     * @return 1=成功, 0=mpp_create 失败, -1=其他失败
     *
     * 初始化步骤（共7步）：
     * 1. 防御式清理：如果之前已初始化，先释放旧资源
     * 2. 根据 video_type 设置 MPP 编码类型
     * 3. 调用 mpp_create() 创建解码器上下文，获取 MPI 接口表
     * 4. 调用 mpp_init() 初始化上下文为解码模式（MPP_CTX_DEC）
     * 5. 获取默认解码配置，修改 split_parse 参数后写回
     * 6. 设置输出超时参数（降低 get_frame 空转带来的 CPU 抖动）
     * 7. 预创建 MppPacket 对象（避免每帧重复 malloc/free）
     */
    int Init(int video_type, int fps, void *userdata, int id);

    /**
     * @brief 注册解码完成后的帧回调函数
     * @param callback 帧回调函数对象（std::function）
     * @return 0=成功
     */
    int SetCallback(MppDecoderFrameCallback callback);

    /**
     * @brief 解码视频数据包
     * @param pkt_data 输入码流数据指针（H.264/H.265 Annex B 格式）
     * @param pkt_size 输入码流数据大小（字节）
     * @param pkt_eos  是否为流结束标志（1=是，解码器将排空内部缓冲帧）
     * @return true=至少解码出一帧, false=未解码出帧或出错
     *
     * 解码循环策略：
     * - 先尝试 put_packet（有超时重试），再拉取 get_frame（有超时重试）
     * - 每轮最多拉取 kMaxDrainPerRound 帧，防止单次调用长时间占用线程
     * - get_frame 返回 TIMEOUT 时自适应等待（前几次 sched_yield，后续 usleep）
     * - 遇到 info-change 帧时自动重新配置 DRM 缓冲区组
     * - EOS 帧抛出时设置 stream_eos 标志，主循环退出
     *
     * @note 该函数非完全线程安全：同一 Decoder 的多个 Decode() 调用须外部串行化
     */
    int Decode(uint8_t *pkt_data, int pkt_size, int pkt_eos);

    /**
     * @brief 重置解码器内部状态
     * @return 0=成功
     *
     * 调用 mpp_mpi->reset() 重置解码器上下文，
     * 清除内部 DPB（Decoded Picture Buffer）参考帧缓存。
     * 通常在 seek 操作后调用，确保解码新位置时不依赖旧参考帧。
     *
     * @note reset 不会释放外部缓冲区组（frm_grp），仅清除 MPP 内部的 DPB 引用
     */
    int Reset();

private:
    MppParam mpp_param1 = NULL;                        ///< 预留的 MPP 参数指针
    RK_U32 need_split = 0;                             ///< 是否需要 MPP 内部帧分割：0=BSF已做NAL分割，1=MPP自行扫描起始码
    RK_U32 width_mpp = 0;                              ///< 视频宽度（从码流中解析，info-change时更新）
    RK_U32 height_mpp = 0;                             ///< 视频高度（从码流中解析，info-change时更新）
    MppCodingType mpp_type = MPP_VIDEO_CodingUnused;   ///< 视频编码类型（AVC/HEVC/MJPEG）
    size_t packet_size = 2400 * 1300 * 3 / 2;          ///< 数据包缓冲区大小预估值（预留，当前通过外部传入）
    MpiDecLoopData loop_data{};                        ///< 解码循环运行时数据（零初始化）
    MppPacket packet = NULL;                           ///< 预创建的 MppPacket 对象（在 Decode() 中每帧复用）
    MppFrame frame = NULL;                             ///< 解码输出的 MppFrame 对象
    MppDecoderFrameCallback callback;                  ///< 帧回调函数对象
    int fps = -1;                                      ///< 视频帧率（预留，当前未使用）
    unsigned long last_frame_time_ms = 0;              ///< 上一帧的时间戳（毫秒，用于帧率控制，预留）
    void *userdata = NULL;                             ///< 用户自定义数据指针（回调透传）
    int id = 0;                                        ///< 解码器 ID（回调透传）
    std::mutex mtx;                                    ///< 互斥锁，保护多线程并发调用（当前用于保护 Init 和析构）
};
