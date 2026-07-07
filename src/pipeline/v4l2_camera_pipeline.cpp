#include "pipeline/v4l2_camera_pipeline.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <utility>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/videodev2.h>

#include <opencv2/imgproc.hpp>

#include "config/ini_config.h"
#include "pipeline/inference_service.h"
#include "pipeline/detection_stabilizer.h"
#include "utils/thread_runtime.h"

namespace {

// xioctl(): 带 EINTR 重试的 ioctl 封装，防止系统调用被信号中断后直接失败
// - fd: V4L2 设备文件描述符
// - request: ioctl 请求码（如 VIDIOC_S_FMT、VIDIOC_QBUF 等）
// - arg: 请求参数指针
// - 返回值: ioctl 的返回值，0 表示成功
int xioctl(int fd, unsigned long request, void* arg) {
    int rc;
    do { rc = ::ioctl(fd, request, arg); } while (rc < 0 && errno == EINTR); // EINTR 表示被信号中断，需重试
    return rc;
}

}  // namespace

namespace pipeline {

struct V4l2BufferRequeueState {
    std::mutex mutex;
    int fd = -1;
    uint32_t buffer_type = 0;
    uint32_t num_planes = 1;
    std::vector<size_t> lengths;
    bool active = false;

    bool queue(uint32_t index) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!active || fd < 0 || index >= lengths.size()) return false;
        v4l2_buffer buffer{};
        buffer.type = static_cast<v4l2_buf_type>(buffer_type);
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        v4l2_plane planes[VIDEO_MAX_PLANES]{};
        if (buffer_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
            planes[0].length = static_cast<uint32_t>(lengths[index]);
            buffer.m.planes = planes;
            buffer.length = num_planes;
        }
        return xioctl(fd, VIDIOC_QBUF, &buffer) == 0;
    }

    void deactivate() {
        std::lock_guard<std::mutex> lock(mutex);
        active = false;
        fd = -1;
    }
};

namespace {
struct V4l2BufferLease {
    std::shared_ptr<V4l2BufferRequeueState> state;
    uint32_t index = 0;
    V4l2BufferLease(std::shared_ptr<V4l2BufferRequeueState> value, uint32_t buffer_index)
        : state(std::move(value)), index(buffer_index) {}
    V4l2BufferLease(const V4l2BufferLease&) = delete;
    V4l2BufferLease& operator=(const V4l2BufferLease&) = delete;
    ~V4l2BufferLease() { if (state) state->queue(index); }
};
}  // namespace

// 构造函数：保存配置并初始化各成员变量为默认值
V4l2CameraPipeline::V4l2CameraPipeline(V4l2CameraConfig config) : config_(std::move(config)) {}

// 析构函数：自动调用 stop() 确保设备资源被正确释放
V4l2CameraPipeline::~V4l2CameraPipeline() { stop(); }

// loadFromIni(): 从 INI 配置文件加载 IMX415 摄像头参数
// - path: INI 配置文件路径
// - out: 输出参数，填充解析后的 V4l2CameraConfig 结构体
// - 返回值: true 表示加载成功
// - 解析的字段包括：使能开关、设备路径、分辨率、帧率、像素格式、色彩空间、旋转角度、推理间隔
bool V4l2CameraPipeline::loadFromIni(const std::string& path, V4l2CameraConfig& out) {
    IniConfig cfg;
    if (!cfg.load(path)) return false;                                      // 加载 INI 文件失败
    out.enable = cfg.getBool("video_source_imx415", "enable", false);       // 是否启用该摄像头
    out.device = cfg.getString("video_source_imx415", "device", "/dev/video0"); // V4L2 设备节点路径
    out.width = cfg.getInt("video_source_imx415", "width", 1920);           // 采集宽度
    out.height = cfg.getInt("video_source_imx415", "height", 1080);         // 采集高度
    out.fps = cfg.getInt("video_source_imx415", "fps", 30);                 // 目标帧率
    out.processing_width = std::max(2, cfg.getInt(                          // 处理宽度（须为偶数，GPU 纹理对齐要求）
        "video_source_imx415", "processing_width", out.processing_width)) & ~1;
    out.processing_height = std::max(2, cfg.getInt(                         // 处理高度（须为偶数）
        "video_source_imx415", "processing_height", out.processing_height)) & ~1;
    out.pixel_format = cfg.getString("video_source_imx415", "pixel_format", "NV12"); // 像素格式（NV12/NV21/YUYV/UYVY）
    out.color_space = cfg.getString(                                        // 色彩空间设定（auto/bt601_full/bt601_limited）
        "video_source_imx415", "color_space", "auto");
    out.rotation = cfg.getInt("video_source_imx415", "rotation", 0);       // 图像旋转角度（0/90/180/270）
    if (out.rotation != 0 && out.rotation != 90 &&                          // 校验旋转角度合法性
        out.rotation != 180 && out.rotation != 270) {
        std::cerr << "[IMX415] invalid rotation=" << out.rotation << ", fallback to 0\n";
        out.rotation = 0;                                                   // 非法值回退为 0（不旋转）
    }
    out.inference_interval_ms = cfg.getInt("video_source_imx415", "inference_interval_ms", 200); // 推理间隔（毫秒）
    return true;
}

// openDevice(): 完整的 V4L2 设备打开与初始化流程
// 步骤：打开设备 -> 查询能力 -> 协商格式 -> 配置帧率 -> 初始化 RGA -> 申请缓冲区 -> 入队 -> 开启流
// 返回值: true 表示设备成功打开并开始采集
bool V4l2CameraPipeline::openDevice() {
    // 第1步：以非阻塞读写模式打开 V4L2 设备节点
    fd_ = ::open(config_.device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd_ < 0) {
        std::cerr << "[IMX415] open " << config_.device << " failed: " << std::strerror(errno) << '\n';
        return false;
    }
    // 第2步：查询设备能力（VIDIOC_QUERYCAP），确认是否为视频采集设备
    v4l2_capability cap{};
    if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        std::cerr << "[IMX415] VIDIOC_QUERYCAP failed: " << std::strerror(errno) << '\n';
        closeDevice(); return false;
    }
    // 优先使用设备专用能力位，回退到通用能力位
    const uint32_t caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS)
        ? cap.device_caps : cap.capabilities;
    // 必须支持流式 I/O（V4L2_CAP_STREAMING），否则无法使用 MMAP 方式采集
    if (!(caps & V4L2_CAP_STREAMING)) {
        std::cerr << "[IMX415] node is not a streaming capture device\n";
        closeDevice(); return false;
    }
    // 第3步：确定缓冲区类型（单平面 / 多平面）
    if (caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) {
        buffer_type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;  // 多平面模式（Y/UV 分离）
    } else if (caps & V4L2_CAP_VIDEO_CAPTURE) {
        buffer_type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;          // 单平面模式（Y/UV 交错）
    } else {
        std::cerr << "[IMX415] node has neither capture nor multiplanar capture capability\n";
        closeDevice(); return false;
    }
    // 第4步：将配置中的字符串像素格式映射为 V4L2 FourCC 码
    if (config_.pixel_format == "YUYV") fourcc_ = V4L2_PIX_FMT_YUYV;      // YUYV 4:2:2
    else if (config_.pixel_format == "UYVY") fourcc_ = V4L2_PIX_FMT_UYVY;  // UYVY 4:2:2
    else if (config_.pixel_format == "NV21") fourcc_ = V4L2_PIX_FMT_NV21;  // NV21 YUV420 semi-planar (Y/VU)
    else fourcc_ = V4L2_PIX_FMT_NV12;                                      // 默认 NV12 YUV420 semi-planar (Y/UV)
    // 第5步：设置视频格式（VIDIOC_S_FMT），协商实际采集参数
    v4l2_format format{};
    format.type = static_cast<v4l2_buf_type>(buffer_type_);
    if (buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
        // 多平面格式参数：宽度、高度、像素格式、field 模式
        format.fmt.pix_mp.width = static_cast<uint32_t>(config_.width);
        format.fmt.pix_mp.height = static_cast<uint32_t>(config_.height);
        format.fmt.pix_mp.pixelformat = fourcc_;
        format.fmt.pix_mp.field = V4L2_FIELD_ANY;  // 由驱动决定 field 类型
    } else {
        // 单平面格式参数
        format.fmt.pix.width = static_cast<uint32_t>(config_.width);
        format.fmt.pix.height = static_cast<uint32_t>(config_.height);
        format.fmt.pix.pixelformat = fourcc_;
        format.fmt.pix.field = V4L2_FIELD_ANY;
    }
    if (xioctl(fd_, VIDIOC_S_FMT, &format) < 0) {
        std::cerr << "[IMX415] VIDIOC_S_FMT failed: " << std::strerror(errno) << '\n';
        closeDevice(); return false;
    }
    // 第6步：读回驱动实际协商后的参数（可能与请求值不同）
    if (buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
        config_.width = static_cast<int>(format.fmt.pix_mp.width);          // 实际宽度
        config_.height = static_cast<int>(format.fmt.pix_mp.height);        // 实际高度
        fourcc_ = format.fmt.pix_mp.pixelformat;                            // 实际像素格式
        num_planes_ = format.fmt.pix_mp.num_planes;                         // 平面数量
        stride_ = static_cast<int>(format.fmt.pix_mp.plane_fmt[0].bytesperline); // 行跨度（含 padding）
        size_image_ = format.fmt.pix_mp.plane_fmt[0].sizeimage;             // 单帧图像缓冲区大小
    } else {
        config_.width = static_cast<int>(format.fmt.pix.width);
        config_.height = static_cast<int>(format.fmt.pix.height);
        fourcc_ = format.fmt.pix.pixelformat;
        num_planes_ = 1;                                                    // 单平面固定为 1
        stride_ = static_cast<int>(format.fmt.pix.bytesperline);
        size_image_ = format.fmt.pix.sizeimage;
    }
    // 第7步：推导实际高度跨度（height stride），用于 NV12/NV21 格式的 UV 平面偏移计算
    height_stride_ = config_.height;  // 默认等于图像高度
    if ((fourcc_ == V4L2_PIX_FMT_NV12 || fourcc_ == V4L2_PIX_FMT_NV21) &&
        stride_ > 0 && size_image_ >= static_cast<size_t>(stride_) * config_.height * 3 / 2) {
        // 从缓冲区总大小反推实际的行对齐高度（驱动可能向上对齐到 16 的倍数）
        size_t candidate = size_image_ * 2 / (static_cast<size_t>(stride_) * 3);
        if (candidate > static_cast<size_t>(config_.height)) candidate &= ~size_t{15}; // 向下对齐到 16 的倍数
        if (candidate >= static_cast<size_t>(config_.height) && candidate <= 8192) {
            height_stride_ = static_cast<int>(candidate);  // 采用反推出的对齐高度
        }
    }
    // DMA直达拼接主链只接受单平面NV12/NV21。打包YUV422无法安全伪装成NV12，
    // 必须在启动阶段失败，避免运行后出现花屏或RGA越界。
    if (num_planes_ != 1 ||
        (fourcc_ != V4L2_PIX_FMT_NV12 && fourcc_ != V4L2_PIX_FMT_NV21)) {
        std::cerr << "[IMX415] unsupported negotiated format or plane count: fourcc=0x"
                  << std::hex << fourcc_ << std::dec << " planes=" << num_planes_ << '\n';
        closeDevice(); return false;
    }
    std::cout << "[IMX415] negotiated " << config_.width << 'x' << config_.height
              << " fourcc=" << static_cast<char>(fourcc_ & 0xff)
              << static_cast<char>((fourcc_ >> 8) & 0xff)
              << static_cast<char>((fourcc_ >> 16) & 0xff)
              << static_cast<char>((fourcc_ >> 24) & 0xff)
              << " planes=" << num_planes_
              << " stride=" << stride_ << 'x' << height_stride_
              << " sizeimage=" << size_image_
              << " api=" << (buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE ? "mplane" : "single")
              << '\n';

    // 第9步：设置采集帧率（VIDIOC_S_PARM），配置 timeperframe
    if (config_.fps > 0) {
        v4l2_streamparm parm{};
        parm.type = static_cast<v4l2_buf_type>(buffer_type_);
        parm.parm.capture.timeperframe.numerator = 1;                       // 分子固定为 1
        parm.parm.capture.timeperframe.denominator = static_cast<uint32_t>(config_.fps); // 分母为目标帧率
        if (xioctl(fd_, VIDIOC_S_PARM, &parm) < 0 && errno != EINVAL && errno != ENOTTY) {
            std::cerr << "[IMX415] VIDIOC_S_PARM warning: " << std::strerror(errno) << '\n';
        }
    }

    // 第10步：初始化 RGA 预处理模块（用于 NV12->BGR 硬件颜色空间转换）
    RgaPreprocessConfig rga_config;
    rga_config.use_rga = true;                                              // 启用 RGA 硬件加速
    rga_config.strict_hardware = true;                                      // 严格要求硬件支持
    rga_config.src_format = fourcc_ == V4L2_PIX_FMT_NV21                    // 源格式：NV21 或 NV12
        ? RgaPixelFormat::NV21 : RgaPixelFormat::NV12;
    rga_config.dst_format = RgaPixelFormat::BGR888;                         // 目标格式：BGR888（OpenCV 默认格式）
    rga_config.target_width = std::min(config_.width, config_.processing_width);   // 目标宽度（取较小值）
    rga_config.target_height = std::min(config_.height, config_.processing_height); // 目标高度
    // 从协商后的 V4L2 格式中提取色彩空间、YCbCr 编码、量化范围信息
    const uint32_t negotiated_colorspace = buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
        ? format.fmt.pix_mp.colorspace : format.fmt.pix.colorspace;
    const uint32_t negotiated_ycbcr = buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
        ? format.fmt.pix_mp.ycbcr_enc : format.fmt.pix.ycbcr_enc;
    const uint32_t negotiated_quantization = buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
        ? format.fmt.pix_mp.quantization : format.fmt.pix.quantization;
    // 根据用户配置或驱动协商结果选择 RGA 色彩空间
    if (config_.color_space == "bt601_full") {
        rga_config.color_space = RgaColorSpace::Bt601Full;                 // BT.601 全范围（0-255）
    } else if (config_.color_space == "bt601_limited") {
        rga_config.color_space = RgaColorSpace::Bt601Limited;              // BT.601 限制范围（16-235）
    } else if (config_.color_space == "auto" &&                            // 自动模式：根据驱动协商值推断
               (negotiated_ycbcr == V4L2_YCBCR_ENC_601 ||
                negotiated_colorspace == V4L2_COLORSPACE_SMPTE170M ||
                negotiated_colorspace == V4L2_COLORSPACE_JPEG)) {
        rga_config.color_space = negotiated_quantization == V4L2_QUANTIZATION_FULL_RANGE
            ? RgaColorSpace::Bt601Full : RgaColorSpace::Bt601Limited;
    } else {
        rga_config.color_space = RgaColorSpace::Bt709Limited;              // 默认 BT.709 限制范围
    }
    if (rga_config.color_space == RgaColorSpace::Bt601Full) {
        dma_color_space_ = DmaColorSpace::Bt601Full;
    } else if (rga_config.color_space == RgaColorSpace::Bt601Limited) {
        dma_color_space_ = DmaColorSpace::Bt601Limited;
    } else {
        dma_color_space_ = DmaColorSpace::Bt709Limited;
    }
    std::cout << "[IMX415] colorspace=" << negotiated_colorspace
              << " ycbcr=" << negotiated_ycbcr
              << " quantization=" << negotiated_quantization
              << " configured=" << config_.color_space
              << " processing=" << rga_config.target_width << 'x'
              << rga_config.target_height << '\n';
    // 初始化 RGA 硬件：仅 NV12/NV21 格式可用，且必须成功初始化
    camera_rga_ready_ = (fourcc_ == V4L2_PIX_FMT_NV12 || fourcc_ == V4L2_PIX_FMT_NV21) &&
                        camera_rga_.initialize(rga_config) &&
                        camera_rga_.isRgaActive();

    // 第11步：申请 MMAP 缓冲区（VIDIOC_REQBUFS），用于内核-用户空间零拷贝
    v4l2_requestbuffers request{};
    // 同步队列、最新帧槽位、编码队列和推理预处理会同时短暂持有DMA租约。
    // 8个缓冲只剩1个可供ISP采集，实测会把30fps压到约25fps；请求12个保留余量。
    request.count = 12;
    request.type = static_cast<v4l2_buf_type>(buffer_type_);
    request.memory = V4L2_MEMORY_MMAP;   // 使用内存映射方式
    if (xioctl(fd_, VIDIOC_REQBUFS, &request) < 0 || request.count < 4) {
        std::cerr << "[IMX415] DMA distribution requires at least 4 capture buffers\n";
        closeDevice(); return false;
    }
    // 第12步：逐个查询、映射、导出、入队每个缓冲区
    buffers_.resize(request.count);
    requeue_state_ = std::make_shared<V4l2BufferRequeueState>();
    requeue_state_->fd = fd_;
    requeue_state_->buffer_type = buffer_type_;
    requeue_state_->num_planes = num_planes_;
    requeue_state_->lengths.resize(request.count);
    requeue_state_->active = true;
    for (uint32_t i = 0; i < request.count; ++i) {
        v4l2_buffer buffer{};
        buffer.type = request.type; buffer.memory = request.memory; buffer.index = i;
        v4l2_plane planes[VIDEO_MAX_PLANES]{};  // 多平面描述数组
        if (buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
            buffer.m.planes = planes;           // 指向平面数组
            buffer.length = num_planes_;        // 平面数量
        }
        // 查询缓冲区信息（VIDIOC_QUERYBUF）：获取偏移量和长度
        if (xioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0) { closeDevice(); return false; }
        buffers_[i].length = buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
            ? planes[0].length : buffer.length;  // 缓冲区实际长度
        requeue_state_->lengths[i] = buffers_[i].length;
        const off_t offset = buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
            ? static_cast<off_t>(planes[0].m.mem_offset) : static_cast<off_t>(buffer.m.offset);
        // mmap 将内核缓冲区映射到用户空间，实现零拷贝访问
        buffers_[i].address = ::mmap(nullptr, buffers_[i].length, PROT_READ | PROT_WRITE,
                                     MAP_SHARED, fd_, offset);
        if (buffers_[i].address == MAP_FAILED) { buffers_[i].address = nullptr; closeDevice(); return false; }
        // 尝试导出 DMA-BUF 文件描述符（VIDIOC_EXPBUF），供 RGA 硬件零拷贝使用-->从 V4L2 驱动导出 DMA-BUF fd，RGA 可直接访问内核缓冲区，无需 CPU 拷贝
        v4l2_exportbuffer export_buffer{};
        export_buffer.type = request.type;
        export_buffer.index = i;
        export_buffer.plane = 0;              // 导出第 0 个平面
        export_buffer.flags = O_CLOEXEC;      // 子进程执行时自动关闭 fd
        if (xioctl(fd_, VIDIOC_EXPBUF, &export_buffer) == 0) {
            buffers_[i].dma_fd = export_buffer.fd;  // 保存 DMA-BUF fd
        }
        // 将缓冲区加入驱动采集队列（VIDIOC_QBUF），准备接收数据
        if (!queueBuffer(i)) { closeDevice(); return false; }
    }
    // 统计成功导出 DMA-BUF 的缓冲区数量，全部导出则 RGA 可用零拷贝路径
    const size_t exported = static_cast<size_t>(std::count_if(
        buffers_.begin(), buffers_.end(), [](const Buffer& buffer) { return buffer.dma_fd >= 0; }));
    std::cout << "[IMX415] exported DMA-BUF buffers=" << exported << '/' << buffers_.size()
              << " rga_zero_cpu_copy=" << (camera_rga_ready_ && exported == buffers_.size()) << '\n';
    if (exported != buffers_.size()) {
        std::cerr << "[IMX415] DMA distribution requires every V4L2 buffer to export a valid fd\n";
        closeDevice();
        return false;
    }

    // 第13步：开启视频流（VIDIOC_STREAMON），驱动开始填充缓冲区
    v4l2_buf_type type = static_cast<v4l2_buf_type>(buffer_type_);
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) { closeDevice(); return false; }
    online_.store(true);  // 标记设备在线
    return true;
}

// closeDevice(): 关闭设备并释放所有资源（与 openDevice 的流程反向）
void V4l2CameraPipeline::closeDevice() {
    if (requeue_state_) requeue_state_->deactivate();
    // 如果设备已开启流，先停止流（VIDIOC_STREAMOFF）
    if (fd_ >= 0 && online_.exchange(false)) {
        v4l2_buf_type type = static_cast<v4l2_buf_type>(buffer_type_);
        xioctl(fd_, VIDIOC_STREAMOFF, &type);
    }
    // 释放所有 MMAP 缓冲区映射
    for (auto& buffer : buffers_) {
        if (buffer.address) ::munmap(buffer.address, buffer.length); // 解除内存映射
        if (buffer.dma_fd >= 0) ::close(buffer.dma_fd);             // 关闭 DMA-BUF 文件描述符
    }
    buffers_.clear();
    // 关闭 V4L2 设备文件描述符
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    requeue_state_.reset();
}

// queueBuffer(): 将指定索引的缓冲区重新加入 V4L2 驱动采集队列（VIDIOC_QBUF）
// - index: 缓冲区在 buffers_ 数组中的索引
// - 返回值: true 表示入队成功
// - 采集线程拿到一帧数据后，处理线程处理完毕需要将该缓冲区重新入队，形成循环
bool V4l2CameraPipeline::queueBuffer(uint32_t index) {
    return requeue_state_ && requeue_state_->queue(index);
}

// start(): 启动摄像头管线，包括打开设备、启动采集/处理/推理三个线程
// - 返回值: true 表示管线成功启动
bool V4l2CameraPipeline::start() {
    if (running_.load()) return true;       // 已在运行，幂等返回
    if (!config_.enable) return false;      // 配置中未启用，直接返回
    if (!openDevice()) return false;        // 打开设备失败
    running_.store(true);                   // 设置运行标志
    // 启动采集线程：负责从 V4L2 驱动取出原始帧数据
    capture_thread_ = std::thread(&V4l2CameraPipeline::captureLoop, this);
    // 启动处理线程：负责颜色转换、旋转、叠加层、推理提交
    processing_thread_ = std::thread(&V4l2CameraPipeline::processingLoop, this);
    // 如果配置了推理服务，启动独立的推理线程（异步执行，不阻塞处理管线）
    if (inference_service_) {
        inference_thread_ = std::thread(&V4l2CameraPipeline::inferenceLoop, this);
    }
    return true;
}

// stop(): 停止管线，依次通知所有线程退出并 join，最后关闭设备
void V4l2CameraPipeline::stop() {
    if (!running_.exchange(false)) { closeDevice(); return; } // 原子地设置停止标志
    latest_cv_.notify_all();                                   // 唤醒可能在等待新帧的处理线程
    inference_cv_.notify_all();                                // 唤醒可能在等待推理任务的推理线程
    if (capture_thread_.joinable()) capture_thread_.join();    // 等待采集线程退出
    if (processing_thread_.joinable()) processing_thread_.join(); // 等待处理线程退出
    if (inference_thread_.joinable()) inference_thread_.join();   // 等待推理线程退出
    closeDevice();                                             // 关闭设备、释放资源
}

// convertFrame(): 将原始 V4L2 帧数据转换为 OpenCV BGR 格式
// - data: 指向原始帧数据的指针（MMAP 映射地址）
// - bytes: 原始数据的字节数
// - bgr: 输出参数，转换后的 BGR888 格式 cv::Mat
// - 返回值: true 表示转换成功
// - 支持 YUYV/UYVY/NV12/NV21 四种输入格式
bool V4l2CameraPipeline::convertFrame(const void* data, size_t bytes, cv::Mat& bgr) const {
    // YUYV 格式转换：Y0 U0 Y1 V0 打包格式，每像素 2 字节
    if (fourcc_ == V4L2_PIX_FMT_YUYV) {
        if (bytes < static_cast<size_t>(config_.height * stride_)) return false;
        cv::Mat yuyv(config_.height, config_.width, CV_8UC2,                     // 双通道无符号8位
                     const_cast<void*>(data), static_cast<size_t>(stride_));     // 不拷贝，原地构造 Mat header
        cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUY2);                        // OpenCV 颜色空间转换
        return true;
    }
    // UYVY 格式转换：U0 Y0 V0 Y1 打包格式
    if (fourcc_ == V4L2_PIX_FMT_UYVY) {
        if (bytes < static_cast<size_t>(config_.height * stride_)) return false;
        cv::Mat uyvy(config_.height, config_.width, CV_8UC2,
                     const_cast<void*>(data), static_cast<size_t>(stride_));
        cv::cvtColor(uyvy, bgr, cv::COLOR_YUV2BGR_UYVY);
        return true;
    }
    // 仅支持 NV12/NV21，其余格式返回失败
    if (fourcc_ != V4L2_PIX_FMT_NV12 && fourcc_ != V4L2_PIX_FMT_NV21) return false;
    const int stride = stride_ > 0 ? stride_ : config_.width;
    if (bytes < static_cast<size_t>(stride * config_.height * 3 / 2)) return false; // 数据不足
    // NV12/NV21 转换：先去除行 padding（stride -> width 紧密排列），再颜色转换
    cv::Mat compact(config_.height * 3 / 2, config_.width, CV_8UC1);               // 紧凑排列的单通道图像
    const auto* source = static_cast<const uint8_t*>(data);
    // 逐行复制，跳过 stride 中的 padding 字节
    for (int row = 0; row < config_.height * 3 / 2; ++row)
        std::memcpy(compact.ptr(row), source + static_cast<size_t>(row * stride), config_.width);
    // OpenCV 颜色空间转换，NV21 和 NV12 的区别仅在于 UV 排列顺序
    cv::cvtColor(compact, bgr, fourcc_ == V4L2_PIX_FMT_NV21
        ? cv::COLOR_YUV2BGR_NV21 : cv::COLOR_YUV2BGR_NV12);
    return true;
}

// captureLoop(): 采集线程主循环
// - 使用 poll() 等待 V4L2 设备有数据可读（500ms 超时）
// - 通过 VIDIOC_DQBUF 取出已填充的缓冲区
// - 将最新的缓冲区索引、字节数、时间戳写入共享变量，通知处理线程
// - 仅保留最新帧：如果上一帧尚未被处理线程取走，则将其直接重新入队（丢弃）
void V4l2CameraPipeline::captureLoop() {
    utils::applyThreadRuntime("imx415_capture", "imx415-cap");  // 设置线程调度策略和名称
    while (running_.load()) {
        // 使用 poll 等待 V4L2 设备文件描述符可读（有帧数据就绪）
        pollfd descriptor{fd_, POLLIN, 0};
        const int ready = ::poll(&descriptor, 1, 500);  // 500ms 超时，保证能响应停止信号
        if (ready <= 0) continue;                        // 超时或出错，继续下一轮

        // 从驱动取出已填充的缓冲区（VIDIOC_DQBUF）
        v4l2_buffer buffer{};
        buffer.type = static_cast<v4l2_buf_type>(buffer_type_);
        buffer.memory = V4L2_MEMORY_MMAP;
        v4l2_plane planes[VIDEO_MAX_PLANES]{};
        if (buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
            buffer.m.planes = planes;
            buffer.length = num_planes_;
        }
        if (xioctl(fd_, VIDIOC_DQBUF, &buffer) < 0) {
            // EAGAIN 表示非阻塞模式下暂无数据，其他错误标记设备离线
            if (errno != EAGAIN) online_.store(false);
            continue;
        }
        // 获取实际填充的数据字节数（多平面取 plane[0]，单平面取 buffer.bytesused）
        const size_t bytes_used = buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
            ? planes[0].bytesused : buffer.bytesused;
        if (buffer.index < buffers_.size()) {
            // 将最新帧信息写入共享变量（加锁保护），供处理线程消费
            std::lock_guard<std::mutex> lock(latest_mutex_);
            // 如果上一帧尚未被处理线程取走（latest_buffer_index_ 仍 >= 0），则将其丢弃并重新入队
            if (latest_buffer_index_ >= 0) {
                queueBuffer(static_cast<uint32_t>(latest_buffer_index_)); // 旧帧重新入队
                dropped_frames_.fetch_add(1);                             // 丢帧计数+1
            }
            // 更新最新帧的缓冲区索引和数据量
            latest_buffer_index_ = static_cast<int>(buffer.index);
            latest_bytes_used_ = bytes_used;
            latest_timestamp_ = std::chrono::system_clock::now();         // 记录到达时间
            // 优先使用 V4L2 硬件时间戳（单调时钟），回退到软件时间戳
            if ((buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) != 0 &&
                (buffer.timestamp.tv_sec != 0 || buffer.timestamp.tv_usec != 0)) {
                // 将 struct timeval 转换为纳秒
                latest_mono_ns_ = static_cast<int64_t>(buffer.timestamp.tv_sec) * 1000000000LL +
                                  static_cast<int64_t>(buffer.timestamp.tv_usec) * 1000LL;
            } else {
                latest_mono_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
            }
            ++latest_seq_;                          // 递增帧序号
            captured_frames_.fetch_add(1);           // 累计采集帧数
            online_.store(true);                     // 标记设备在线
            latest_cv_.notify_one();                 // 通知处理线程有新帧可用
        } else {
            // 缓冲区索引越界（异常情况），直接重新入队
            queueBuffer(buffer.index);
        }
    }
}

// processingLoop(): 处理线程主循环
// - 等待采集线程提供的最新帧数据
// - 执行颜色空间转换（优先 RGA 硬件加速，回退到 CPU 转换）
// - 执行图像旋转
// - 定期提交帧到推理线程
// - 构建叠加层信息并通过回调输出成品帧到 FrameHub
void V4l2CameraPipeline::processingLoop() {
    utils::applyThreadRuntime("imx415_process", "imx415-proc");  // 设置线程调度策略和名称
    uint64_t consumed = 0;                                        // 已消费的帧序号，用于判断是否有新帧
    auto last_inference_submit = std::chrono::steady_clock::time_point{}; // 上次提交推理的时间
    while (running_.load()) {
        cv::Mat frame;
        std::chrono::system_clock::time_point timestamp;
        int64_t capture_mono_ns = 0;
        int buffer_index = -1;
        size_t bytes_used = 0;
        {
            // 等待采集线程通知有新帧可用（500ms 超时）
            std::unique_lock<std::mutex> lock(latest_mutex_);
            latest_cv_.wait_for(lock, std::chrono::milliseconds(500), [&] {
                return !running_.load() || latest_seq_ != consumed;
            });
            if (!running_.load()) break;                 // 收到停止信号，退出
            if (latest_seq_ == consumed || latest_buffer_index_ < 0) continue; // 无新帧，继续等待
            consumed = latest_seq_;                      // 记录已消费的帧序号
            buffer_index = latest_buffer_index_;         // 取出缓冲区索引
            bytes_used = latest_bytes_used_;             // 取出数据字节数
            latest_buffer_index_ = -1;                   // 清空索引，表示已取走最新帧
            timestamp = latest_timestamp_;               // 取出时间戳
            capture_mono_ns = latest_mono_ns_;           // 取出单调时钟纳秒
        }
        if (buffer_index < 0 || static_cast<size_t>(buffer_index) >= buffers_.size()) continue;
        const auto& capture_buffer = buffers_[static_cast<size_t>(buffer_index)];

        const auto now = std::chrono::steady_clock::now();
        const bool should_infer = inference_service_ &&
            (last_inference_submit.time_since_epoch().count() == 0 ||
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 now - last_inference_submit).count() >= config_.inference_interval_ms);

        // 获取缓存的检测结果（来自推理线程），用于叠加到画面上
        std::array<detect_result_group_t, kMaxInferenceModels> detections{};
        bool draw_cached = false;
        int64_t detection_mono_ns = 0;
        {
            std::lock_guard<std::mutex> lock(detection_mutex_);
            if (detection_cache_valid_) {
                for (size_t i = 0; i < cached_detections_.size(); ++i) {
                    const int64_t detected_at = cached_detection_mono_ns_[i];
                    // 仅使用"新鲜"的检测结果：检测时间不晚于采集时间，且不超过 700ms
                    const bool fresh = detected_at > 0 && capture_mono_ns >= detected_at &&
                        capture_mono_ns - detected_at <= 700000000LL;
                    if (!fresh) continue;
                    detections[i] = cached_detections_[i];          // 复制检测结果
                    draw_cached = draw_cached || cached_detections_[i].count > 0; // 是否有检测到目标
                    detection_mono_ns = std::max(detection_mono_ns, detected_at);
                }
            } else {
                detection_cache_valid_ = false;  // 缓存无效，后续帧将触发生效标记重置
            }
        }

        // 预览主链直接发布DMA；只有达到推理采样间隔时才物化一次BGR并旋转。
        if (should_infer) {
            bool converted = camera_rga_ready_ && capture_buffer.dma_fd >= 0 &&
                camera_rga_.processFromFd(capture_buffer.dma_fd, config_.width,
                    config_.height, stride_, frame, height_stride_, capture_buffer.length);
            if (!converted) converted = convertFrame(capture_buffer.address, bytes_used, frame);
            if (converted && config_.rotation != 0) {
                cv::Mat rotated;
                if (!camera_rga_.rotateBgr(frame, rotated, config_.rotation)) {
                    if (config_.rotation == 90) {
                        cv::rotate(frame, rotated, cv::ROTATE_90_CLOCKWISE);
                    } else if (config_.rotation == 180) {
                        cv::rotate(frame, rotated, cv::ROTATE_180);
                    } else {
                        cv::rotate(frame, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
                    }
                }
                frame = std::move(rotated);
            }
            if (converted && !frame.empty()) {
                std::lock_guard<std::mutex> lock(inference_mutex_);
                inference_frame_ = frame;                // 复制帧数据到推理缓冲区
                inference_timestamp_ = timestamp;         // 传递时间戳
                inference_mono_ns_ = capture_mono_ns;     // 传递采集纳秒
                inference_pending_ = true;                // 设置推理待处理标志
                inference_cv_.notify_one();
                last_inference_submit = now;
            }
        }

        // 更新帧率统计（处理帧率）
        frame_rate_.tick();
        // 构建叠加层信息，包含检测框、帧率统计等元数据
        FrameHub::FrameOverlay overlay;
        const int inference_width = std::min(config_.width, config_.processing_width);
        const int inference_height = std::min(config_.height, config_.processing_height);
        overlay.source_width = config_.rotation == 90 || config_.rotation == 270
            ? inference_height : inference_width;
        overlay.source_height = config_.rotation == 90 || config_.rotation == 270
            ? inference_width : inference_height;
        overlay.frame_fps = frame_rate_.rate();           // 处理帧率
        overlay.inference_fps = inference_rate_.rate();   // 推理帧率
        overlay.frames_captured = captured_frames_.load(std::memory_order_relaxed);  // 累计采集帧数
        overlay.frames_dropped = dropped_frames_.load(std::memory_order_relaxed);   // 累计丢帧数
        overlay.detections = detections;                 // 检测结果数组
        overlay.detections_valid = draw_cached;          // 检测结果是否有效
        overlay.detection_mono_ns = detection_mono_ns;   // 检测时间戳

        // 原始NV12/NV21 DMA缓冲直接交给FrameHub。lease析构时才QBUF，确保RGA
        // 合成完成前摄像头驱动不会覆盖当前帧。
        FrameHub::DmaFrame dma_frame;
        dma_frame.fd = capture_buffer.dma_fd;
        dma_frame.width = config_.width;
        dma_frame.height = config_.height;
        dma_frame.width_stride = stride_;
        dma_frame.height_stride = height_stride_;
        dma_frame.buffer_size = capture_buffer.length;
        dma_frame.rotation = config_.rotation;
        dma_frame.format = fourcc_ == V4L2_PIX_FMT_NV21
            ? DmaPixelFormat::NV21 : DmaPixelFormat::NV12;
        dma_frame.color_space = dma_color_space_;
        dma_frame.lease = std::make_shared<V4l2BufferLease>(
            requeue_state_, static_cast<uint32_t>(buffer_index));
        if (dma_frame.valid() && frame_callback_) {
            frame_callback_(dma_frame, timestamp, capture_mono_ns, overlay);
        }
        // 无消费者或发布失败时，局部lease在本轮结束立即归还缓冲。
    }
}

// inferenceLoop(): 推理线程主循环
// - 等待处理线程提交的待推理帧
// - 调用推理服务执行目标检测（COCO 类别 + Fire/Smoke 检测）
// - 对检测结果进行时序稳定化处理，更新检测缓存供处理线程使用
// - 构建推理统计信息并通过回调上报
void V4l2CameraPipeline::inferenceLoop() {
    utils::applyThreadRuntime("imx415_inference", "imx415-infer");  // 设置线程调度策略和名称
    while (running_.load()) {
        cv::Mat frame;
        std::chrono::system_clock::time_point timestamp;
        int64_t capture_mono_ns = 0;
        {
            // 等待处理线程提交新的推理任务（500ms 超时）
            std::unique_lock<std::mutex> lock(inference_mutex_);
            inference_cv_.wait_for(lock, std::chrono::milliseconds(500), [this] {
                return !running_.load() || inference_pending_;
            });
            if (!running_.load()) break;                 // 停止信号，退出
            if (!inference_pending_ || inference_frame_.empty()) continue; // 无待处理任务
            frame = std::move(inference_frame_);         // 转移帧引用，避免成员滞留上一帧内存
            timestamp = inference_timestamp_;             // 取出时间戳
            capture_mono_ns = inference_mono_ns_;         // 取出采集纳秒
            inference_pending_ = false;                   // 清除待处理标志
        }

        // 调用推理服务执行目标检测
        InferenceResult result = inference_service_->infer(frame, 1);
        if (!result.success) continue;                   // 推理失败，跳过本帧

        inference_rate_.tick();                          // 更新推理帧率统计
        data_lifecycle::InferenceStats stats;            // 构建推理统计信息
        {
            std::lock_guard<std::mutex> lock(detection_mutex_);
            // 遍历所有推理模型（最多 kMaxInferenceModels 个），更新检测缓存
            for (size_t i = 0; i < cached_detections_.size(); ++i) {
                if (result.updated_mask & (1u << i)) {   // 该模型槽位有更新
                    // 对检测结果进行时序稳定化（过滤闪烁、平滑框坐标）
                    stabilizeDetections(cached_detections_[i], result.detections[i]);
                    cached_detections_[i] = result.detections[i];       // 更新缓存
                    cached_detection_mono_ns_[i] = capture_mono_ns;     // 记录检测时间
                }
            }
            detection_cache_valid_ = true;               // 标记检测缓存有效
            stats.stream_id = 2;                         // 流 ID 标识（IMX415 摄像头）
            // B路复用COCO上下文slot 0，slot 1为Fire/Smoke。
            // 统计 COCO 类别（slot 0）：人物、猫、狗、其他
            {
                const auto& coco = cached_detections_[0];
                int persons = 0, cats = 0, dogs = 0, others = 0;
                for (int i = 0; i < coco.count; ++i) {
                    switch (coco.results[i].class_id) {
                        case 0: ++persons; break;        // COCO class_id=0: person
                        case 15: ++cats; break;          // COCO class_id=15: cat
                        case 16: ++dogs; break;          // COCO class_id=16: dog
                        default: ++others; break;        // 其他类别
                    }
                }
                stats.person_count = persons;            // 检测到的人数
                stats.cat_count = cats;                  // 检测到的猫数
                stats.dog_count = dogs;                  // 检测到的狗数
            }
            // 统计 Fire/Smoke 检测结果（slot 1）
            {
                const auto& fire = cached_detections_[1];
                int fires = 0, smokes = 0;
                for (int i = 0; i < fire.count; ++i) {
                    if (fire.results[i].class_id == 1) ++fires;    // class_id=1: fire（火焰）
                    else if (fire.results[i].class_id == 0) ++smokes; // class_id=0: smoke（烟雾）
                }
                stats.fire_count = fires;                // 检测到的火焰数
                stats.smoke_count = smokes;              // 检测到的烟雾数
            }
        }
        // 填充时间戳信息
        stats.ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            timestamp.time_since_epoch()).count();       // 系统时钟毫秒时间戳
        stats.capture_mono_ns = capture_mono_ns;          // 单调时钟纳秒
        stats.infer_executed = true;                      // 标记推理已执行

        // 调用推理回调函数，上报统计信息
        if (inference_callback_) inference_callback_(stats);
    }
}
}  // namespace pipeline
