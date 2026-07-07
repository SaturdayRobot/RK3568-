/**
 * @file rknnPool.hpp
 * @brief RKNN模型推理池实现
 * 
 * 该文件实现了RKNN模型的加载、初始化和推理功能，
 * 为目标检测任务提供模型推理支持。
 */

#ifndef _rknnPool_H
#define _rknnPool_H

#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <mutex>
#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "rknn_api.h"
#include "data_processing/postprocess.h"
#include "pipeline/rga_preprocessor.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"

using cv::Mat;
using std::vector;

/**
 * @brief 从文件中加载数据
 * @param fp 文件指针
 * @param ofst 偏移量
 * @param sz 数据大小
 * @return 加载的数据
 */
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);

/**
 * @brief 加载模型文件
 * @param filename 模型文件名
 * @param model_size 模型大小
 * @return 加载的模型数据
 */
static unsigned char *load_model(const char *filename, int *model_size);

/**
 * @class rknn_lite
 * @brief RKNN模型轻量级封装类
 * 
 * 封装了RKNN模型的加载、初始化和推理功能
 * 提供简洁的接口用于目标检测任务
 */
class rknn_lite
{
private:
    rknn_context rkModel = 0;        // RKNN模型上下文
    unsigned char *model_data = nullptr;       // 模型数据
    rknn_sdk_version version;        // RKNN SDK版本
    rknn_input_output_num io_num;    // 输入输出数量
    rknn_tensor_attr *input_attrs = nullptr;   // 输入属性
    rknn_tensor_attr *output_attrs = nullptr;  // 输出属性
    rknn_input inputs[1];            // 输入数据
    int ret = 0;                     // 返回值
    int channel = 3;                 // 通道数
    int width = 0;                   // 宽度
    int height = 0;                  // 高度
    int class_num = 0;               // 类别数
    int id;                          // 模型ID
    std::string label_path_;        // 标签文件路径（每个模型独立）
    float confidence_threshold_ = 0.40F;
    float nms_threshold_ = 0.50F;
    pipeline::RgaPreprocessor rga_preproc_;  // RGA 硬件加速预处理器
    pipeline::RgaPreprocessor rga_fd_preproc_;  // DMA-FD 直通预处理器(NV12->RGB)
    bool fd_hw_path_ready_ = false;  // FD硬件直通链路是否就绪（严格硬件模式）
    bool rga_preproc_skip_ = false; // 本次推理跳过RGA（使用共享预处理）
    rknn_core_mask core_mask_ = RKNN_NPU_CORE_AUTO; // 当前 NPU 核心掩码
    cv::Mat rga_output_buf_;  // 预分配的RGA输出缓冲区，避免每帧堆分配
    cv::Mat letterbox_resized_buf_;
    int mat_preproc_width_ = 0;
    int mat_preproc_height_ = 0;
    rknn_tensor_mem* input_mem_ = nullptr;      // RKNN 分配、RGA/NPU 共享的输入 DMA-BUF
    rknn_tensor_attr bound_input_attr_{};       // rknn_set_io_mem 实际绑定的外部输入属性
    rknn_tensor_attr native_input_attr_{};      // NPU 原生输入属性（用于诊断/约束检查）
    int input_wstride_ = 0;                     // RGB 输入像素步长（不是字节数）
    int input_hstride_ = 0;
    bool zero_copy_input_ready_ = false;
    bool prepared_input_ready_ = false;
    float prepared_scale_w_ = 1.0F;
    float prepared_scale_h_ = 1.0F;
    float prepared_pad_x_ = 0.0F;
    float prepared_pad_y_ = 0.0F;
    bool fd_padding_valid_ = false;
    int fd_last_resized_w_ = 0;
    int fd_last_resized_h_ = 0;
    int fd_last_offset_x_ = 0;
    int fd_last_offset_y_ = 0;

    bool preprocessMatLetterbox(const cv::Mat& source,
                                float& scale_w, float& scale_h,
                                float& pad_x, float& pad_y);
    bool initializeBoundInput();
    bool copyMatToBoundInput(const cv::Mat& source);
    bool validateOutputLayout() const;

public:
    Mat ori_img;                     // 原始图像
    
    /**
     * @brief 推理函数
     * @param detect_result_group 检测结果组
     * @return 推理结果
     */
    int interf(detect_result_group_t &detect_result_group, bool draw_result = true,
               std::mutex* npu_mutex = nullptr);
    // 两阶段 FD 推理：先完成所有模型的 RGA 读取，再释放源帧，最后串行运行 NPU。
    int prepareFromFd(int src_fd, int src_width, int src_height,
                      int src_stride, int src_height_stride,
                      size_t src_buffer_size,
                      pipeline::RgaPixelFormat src_format = pipeline::RgaPixelFormat::NV12);
    int inferPrepared(detect_result_group_t &detect_result_group,
                      std::mutex* npu_mutex = nullptr);

    /**
     * @brief 设置共享预处理缓冲区（同源多模型跳过重复RGA）
     * @param buf 已经过RGA预处理的RGB 640x640图像
     */
    bool setSharedPreproc(const cv::Mat& buf) {
        rga_preproc_skip_ = false;
        if (buf.empty() || buf.type() != CV_8UC3 ||
            buf.cols != width || buf.rows != height) {
            return false;
        }
        buf.copyTo(rga_output_buf_);
        rga_preproc_skip_ = true;
        return true;
    }
    const cv::Mat& getPreprocessedBuf() const { return rga_output_buf_; }

    /**
     * @brief 动态切换 NPU 核心掩码
     * @param mask 新的核心掩码（如 RKNN_NPU_CORE_0）
     * @return true 切换成功
     *
     * 由热管理器在温度变化时调用，实现动态降级/恢复。
     */
    bool setCoreMask(rknn_core_mask mask);

    /**
     * @brief 获取当前核心掩码
     */
    rknn_core_mask getCoreMask() const { return core_mask_; }

    /**
     * @brief 检查模型是否初始化成功
     * @return true 模型可用于推理
     *
     * rknn_init 失败时 input_attrs 会被设为 nullptr，
     * 此方法供上层（如 InferenceService）在构造后验证有效性。
     */
    bool isValid() const { return rkModel != 0 && input_attrs != nullptr && output_attrs != nullptr; }
    bool supportsFdInput() const { return isValid() && fd_hw_path_ready_ && zero_copy_input_ready_; }

    /**
     * @brief 构造函数
     * @param dst 模型路径
     * @param n 预留参数
     * @param class_num 类别数
     * @param id 模型ID
     * @param core_mask NPU 核心掩码（默认 AUTO）
     * @param label_path 标签文件路径（空字符串则使用默认路径）
     */
    rknn_lite(char *dst, int n, int class_num, int id,
              rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO,
              const std::string& label_path = "",
              float confidence_threshold = 0.40F,
              float nms_threshold = 0.50F);
    
    /**
     * @brief 析构函数
     */
    ~rknn_lite();
};

/**
 * @brief 构造函数实现->实例的初始化
 * @param model_name 模型文件名
 * @param n 预留参数
 * @param class_num 类别数
 * @param id 模型ID
 */
rknn_lite::rknn_lite(char *model_name, int n, int class_num, int id,
                    rknn_core_mask core_mask, const std::string& label_path,
                    float confidence_threshold, float nms_threshold)
{
    (void)n; // 预留参数，当前版本未使用
    // 1：初始化类别数、模型ID 和标签路径
    this->class_num = class_num;
    this->id = id;
    this->core_mask_ = core_mask;
    this->label_path_ = label_path;
    this->confidence_threshold_ = std::clamp(confidence_threshold, 0.05F, 0.95F);
    this->nms_threshold_ = std::clamp(nms_threshold, 0.10F, 0.90F);
    // 模型加载入口就在构造函数：load_model -> rknn_init
    /* Create the neural network */
    printf("Loading model id = %d\n", id);
    int model_data_size = 0;
    // 2：加载模型文件数据
    model_data = load_model(model_name, &model_data_size);
    if (!model_data || model_data_size <= 0)
    {
        printf("load_model failed\n");
        return;
    }
    // 3：通过模型文件初始化rknn对象
    ret = rknn_init(&rkModel, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        // 标记初始化失败，由调用方检查
        input_attrs = nullptr;
        output_attrs = nullptr;
        return;
    }
    // 4：设置 NPU 核心掩码（由调用方指定，支持热管理动态切换）
    // RK3568 为单核 NPU，旧版 runtime 对显式 CORE_0 返回 -13。AUTO 就是该平台
    // 的正确模式，保持 runtime 默认设置即可，不做一次必然失败的控制调用。
    if (core_mask_ != RKNN_NPU_CORE_AUTO)
    {
        ret = rknn_set_core_mask(rkModel, core_mask_);
        if (ret < 0)
        {
            printf("rknn_set_core_mask(%d) error ret=%d, fallback to AUTO\n", core_mask_, ret);
            core_mask_ = RKNN_NPU_CORE_AUTO;
        }
    }

    // 5：查询SDK版本
    ret = rknn_query(rkModel, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_query SDK_VERSION error ret=%d\n", ret);
        // 版本查询失败不致命，继续
    }
    // 获取模型的输入参数
    ret = rknn_query(rkModel, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query IN_OUT_NUM error ret=%d\n", ret);
        input_attrs = nullptr;
        output_attrs = nullptr;
        return;
    }
    if (io_num.n_input != 1 || io_num.n_output == 0)
    {
        fprintf(stderr, "[rknn_lite] model id=%d: unsupported IO count input=%u output=%u\n",
                id, io_num.n_input, io_num.n_output);
        rknn_destroy(rkModel);
        rkModel = 0;
        input_attrs = nullptr;
        output_attrs = nullptr;
        return;
    }

    // 6：查询输入张量属性
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    memset(input_attrs, 0, sizeof(rknn_tensor_attr) * io_num.n_input);
    for (uint32_t i = 0; i < io_num.n_input; i++)//查询每个输入张量的属性
    {
        input_attrs[i].index = i;
        ret = rknn_query(rkModel, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query INPUT_ATTR[%d] error ret=%d\n", i, ret);
            delete[] input_attrs;
            input_attrs = nullptr;
            output_attrs = nullptr;
            if (rkModel)
            {
                rknn_destroy(rkModel);
                rkModel = 0;
            }
            return;
        }
    }

    // 7：查询输出张量属性
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, sizeof(rknn_tensor_attr) * io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;//设置输出张量的索引
        ret = rknn_query(rkModel, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query OUTPUT_ATTR[%d] error ret=%d\n", i, ret);
            delete[] input_attrs;
            delete[] output_attrs;
            input_attrs = nullptr;
            output_attrs = nullptr;
            if (rkModel)
            {
                rknn_destroy(rkModel);
                rkModel = 0;
            }
            return;
        }
    }

    // 8：输入维度解析
    if (input_attrs[0].n_dims != 4 || input_attrs[0].dims[0] != 1)
    {
        fprintf(stderr, "[rknn_lite] model id=%d: only static batch-1 4D input is supported\n", id);
        delete[] input_attrs;
        delete[] output_attrs;
        input_attrs = nullptr;
        output_attrs = nullptr;
        rknn_destroy(rkModel);
        rkModel = 0;
        return;
    }
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        channel = input_attrs[0].dims[1];//获取通道数
        height = input_attrs[0].dims[2];//获取高度
        width = input_attrs[0].dims[3];//获取宽度
    }
    else
    {
        height = input_attrs[0].dims[1];//获取高度
        width = input_attrs[0].dims[2];//获取宽度
        channel = input_attrs[0].dims[3];//获取通道数
    }
    if (width <= 0 || height <= 0 || width % 32 != 0 || height % 32 != 0 || channel != 3 ||
        (input_attrs[0].fmt != RKNN_TENSOR_NCHW &&
         input_attrs[0].fmt != RKNN_TENSOR_NHWC))
    {
        fprintf(stderr,
                "[rknn_lite] model id=%d: unsupported input shape/format %dx%dx%d fmt=%d\n",
                id, height, width, channel, input_attrs[0].fmt);
        delete[] input_attrs;
        delete[] output_attrs;
        input_attrs = nullptr;
        output_attrs = nullptr;
        rknn_destroy(rkModel);
        rkModel = 0;
        return;
    }
    if (!validateOutputLayout())
    {
        fprintf(stderr, "[rknn_lite] model id=%d: unsupported output tensor layout\n", id);
        delete[] input_attrs;
        delete[] output_attrs;
        input_attrs = nullptr;
        output_attrs = nullptr;
        rknn_destroy(rkModel);
        rkModel = 0;
        return;
    }
    // 9：输入参数配置
    memset(inputs, 0, sizeof(inputs));//初始化输入数据
    inputs[0].index = 0;//设置输入张量的索引
    inputs[0].type = RKNN_TENSOR_UINT8;//设置输入张量的数据类型为无符号整数
    inputs[0].size = width * height * channel;//设置输入张量的大小
    inputs[0].fmt = RKNN_TENSOR_NHWC;//设置输入张量的格式为NHWC
    inputs[0].pass_through = 0;//设置输入张量是否通过NPU处理，0表示不通过
    // 10：初始化 RGA 硬件加速预处理器（BGR→RGB + resize）
    {
        pipeline::RgaPreprocessConfig rga_cfg;
        rga_cfg.use_rga = true;   // 逐步恢复RGA：仅Mat预处理
        rga_cfg.target_width = width;
        rga_cfg.target_height = height;
        rga_cfg.src_format = pipeline::RgaPixelFormat::BGR888;
        rga_cfg.dst_format = pipeline::RgaPixelFormat::RGB888;
        rga_preproc_.initialize(rga_cfg);

        pipeline::RgaPreprocessConfig rga_fd_cfg;
        rga_fd_cfg.use_rga = true;
        rga_fd_cfg.strict_hardware = true;
        rga_fd_cfg.target_width = width;
        rga_fd_cfg.target_height = height;
        rga_fd_cfg.src_format = pipeline::RgaPixelFormat::NV12;
        rga_fd_cfg.dst_format = pipeline::RgaPixelFormat::RGB888;
        rga_fd_preproc_.initialize(rga_fd_cfg);

        // strict_hardware=true 时，RGA不可用就意味着FD路径不可用
        fd_hw_path_ready_ = rga_fd_preproc_.isRgaActive();
        if (!fd_hw_path_ready_)
        {
            fprintf(stderr, "[rknn_lite] model id=%d: FD hardware path is not ready.\n", id);
        }
    }

    // 预分配RGA输出缓冲区（模型输入尺寸），避免每帧堆分配
    rga_output_buf_.create(height, width, CV_8UC3);

    // 绑定一块由 RKNN 分配的 DMA-BUF。绑定失败不影响传统 Mat 路径，但会禁用
    // FD 直通，避免悄悄回退到带 CPU 拷贝的伪零拷贝路径。
    zero_copy_input_ready_ = initializeBoundInput();
    fd_hw_path_ready_ = fd_hw_path_ready_ && zero_copy_input_ready_;
    if (!zero_copy_input_ready_)
    {
        fprintf(stderr, "[rknn_lite] model id=%d: RKNN bound input unavailable; FD path disabled.\n", id);
    }
}

bool rknn_lite::validateOutputLayout() const
{
    if (!output_attrs || width <= 0 || height <= 0) return false;
    auto is_int8_nchw = [](const rknn_tensor_attr& attr) {
        return attr.type == RKNN_TENSOR_INT8 && attr.fmt == RKNN_TENSOR_NCHW &&
               attr.n_dims == 4 && attr.dims[0] == 1;
    };

    if (io_num.n_output < 6 || io_num.n_output % 3 != 0) return false;
    const uint32_t per_branch = io_num.n_output / 3;
    if (per_branch != 2 && per_branch != 3) return false;
    for (uint32_t branch = 0; branch < 3; ++branch)
    {
        const auto& box = output_attrs[branch * per_branch];
        const auto& score = output_attrs[branch * per_branch + 1];
        if (!is_int8_nchw(box) || !is_int8_nchw(score) || box.dims[1] % 4 != 0 ||
            static_cast<int>(score.dims[1]) < class_num ||
            box.dims[2] != score.dims[2] || box.dims[3] != score.dims[3]) return false;
        if (per_branch == 3)
        {
            const auto& sum = output_attrs[branch * per_branch + 2];
            if (!is_int8_nchw(sum) || sum.dims[1] != 1 ||
                sum.dims[2] != box.dims[2] || sum.dims[3] != box.dims[3]) return false;
        }
    }
    return true;
}

bool rknn_lite::initializeBoundInput()
{
    if (rkModel == 0 || input_attrs == nullptr || io_num.n_input != 1 ||
        width <= 0 || height <= 0 || channel != 3)
    {
        return false;
    }

    memset(&native_input_attr_, 0, sizeof(native_input_attr_));
    native_input_attr_.index = 0;
    const int native_ret = rknn_query(rkModel, RKNN_QUERY_NATIVE_INPUT_ATTR,
                                      &native_input_attr_, sizeof(native_input_attr_));
    if (native_ret < 0)
    {
        // 部分旧版 RK3568 runtime 不支持 native attr 查询；外部 UINT8/NHWC
        // 绑定仍可依据普通 INPUT_ATTR 工作。
        memset(&native_input_attr_, 0, sizeof(native_input_attr_));
    }
    else
    {
        fprintf(stderr,
                "[rknn_lite] model id=%d: native input type=%d fmt=%d stride=%ux%u size=%u\n",
                id, native_input_attr_.type, native_input_attr_.fmt,
                native_input_attr_.w_stride, native_input_attr_.h_stride,
                native_input_attr_.size_with_stride);
    }

    bound_input_attr_ = input_attrs[0];
    bound_input_attr_.index = 0;
    bound_input_attr_.type = RKNN_TENSOR_UINT8;
    bound_input_attr_.fmt = RKNN_TENSOR_NHWC;
    bound_input_attr_.pass_through = 0;  // runtime 按模型属性完成必要的量化/布局转换
    input_wstride_ = input_attrs[0].w_stride > 0
        ? static_cast<int>(input_attrs[0].w_stride) : width;
    input_hstride_ = height;
    if (input_wstride_ < width) return false;

    const uint64_t required_size = static_cast<uint64_t>(input_wstride_) *
                                   input_hstride_ * channel;
    if (required_size == 0 || required_size > std::numeric_limits<uint32_t>::max())
    {
        return false;
    }
    const uint32_t allocation_size = input_attrs[0].size_with_stride > 0
        ? input_attrs[0].size_with_stride : input_attrs[0].size;
    // RKNN 官方零拷贝接口要求使用查询得到的 size_with_stride。若它不足以
    // 容纳 UINT8/NHWC RGB 布局，说明该 runtime/model 组合不能安全走此路径。
    if (allocation_size < required_size) return false;
    bound_input_attr_.w_stride = static_cast<uint32_t>(input_wstride_);
    bound_input_attr_.h_stride = static_cast<uint32_t>(input_hstride_);
    bound_input_attr_.size = static_cast<uint32_t>(
        static_cast<uint64_t>(width) * height * channel);
    bound_input_attr_.size_with_stride = allocation_size;

    input_mem_ = rknn_create_mem(rkModel, allocation_size);
    if (!input_mem_ || input_mem_->fd < 0 || input_mem_->size < required_size)
    {
        if (input_mem_) rknn_destroy_mem(rkModel, input_mem_);
        input_mem_ = nullptr;
        return false;
    }
    ret = rknn_set_io_mem(rkModel, input_mem_, &bound_input_attr_);
    if (ret < 0)
    {
        rknn_destroy_mem(rkModel, input_mem_);
        input_mem_ = nullptr;
        return false;
    }

    fprintf(stderr,
            "[rknn_lite] model id=%d: input DMA bound %dx%dx%d stride=%dx%d size=%u fd=%d\n",
            id, width, height, channel, input_wstride_, input_hstride_,
            input_mem_->size, input_mem_->fd);
    return true;
}

bool rknn_lite::copyMatToBoundInput(const cv::Mat& source)
{
    if (!zero_copy_input_ready_ || !input_mem_ || !input_mem_->virt_addr ||
        source.empty() || source.type() != CV_8UC3 ||
        source.cols != width || source.rows != height)
    {
        return false;
    }
    const size_t row_bytes = static_cast<size_t>(width) * channel;
    const size_t target_row_bytes = static_cast<size_t>(input_wstride_) * channel;
    auto* destination = static_cast<uint8_t*>(input_mem_->virt_addr);
    for (int row = 0; row < height; ++row)
    {
        memcpy(destination + static_cast<size_t>(row) * target_row_bytes,
               source.ptr(row), row_bytes);
        if (target_row_bytes > row_bytes)
        {
            memset(destination + static_cast<size_t>(row) * target_row_bytes + row_bytes,
                   114, target_row_bytes - row_bytes);
        }
    }
    fd_padding_valid_ = false; // Mat 内容区域可能与下一帧 FD 的 ROI 不同
    return rknn_mem_sync(rkModel, input_mem_, RKNN_MEMORY_SYNC_TO_DEVICE) >= 0;
}

/**
 * @brief 动态切换 NPU 核心掩码
 */
bool rknn_lite::setCoreMask(rknn_core_mask mask)
{
    if (rkModel == 0) return false;
    if (mask == core_mask_) return true;  // 无变化
    int r = rknn_set_core_mask(rkModel, mask);
    if (r < 0) {
        fprintf(stderr, "[rknn_lite] setCoreMask(%d) failed ret=%d\n", mask, r);
        return false;
    }
    core_mask_ = mask;
    return true;
}

/**
 * @brief 析构函数实现
 * 
 * 释放RKNN模型资源和内存
 */
rknn_lite::~rknn_lite()
{
    if (input_mem_ && rkModel)
    {
        rknn_destroy_mem(rkModel, input_mem_);
        input_mem_ = nullptr;
    }
    if (rkModel)
    {
        ret = rknn_destroy(rkModel);
        (void)ret;
        rkModel = 0;
    }
    if (input_attrs)
    {
        delete[] input_attrs;
        input_attrs = nullptr;
    }
    if (output_attrs)
    {
        delete[] output_attrs;
        output_attrs = nullptr;
    }
    if (model_data)
    {
        free(model_data);
        model_data = nullptr;
    }
}

bool rknn_lite::preprocessMatLetterbox(const cv::Mat& source,
                                       float& scale_w, float& scale_h,
                                       float& pad_x, float& pad_y)
{
    if (source.empty()) return false;
    const float scale = std::min(static_cast<float>(width) / source.cols,
                                 static_cast<float>(height) / source.rows);
    const int resized_w = std::min(width, std::max(2,
        static_cast<int>(std::round(source.cols * scale)) & ~1));
    const int resized_h = std::min(height, std::max(2,
        static_cast<int>(std::round(source.rows * scale)) & ~1));
    if (mat_preproc_width_ != resized_w || mat_preproc_height_ != resized_h) {
        pipeline::RgaPreprocessConfig cfg;
        cfg.use_rga = true;
        cfg.target_width = resized_w;
        cfg.target_height = resized_h;
        cfg.src_format = pipeline::RgaPixelFormat::BGR888;
        cfg.dst_format = pipeline::RgaPixelFormat::RGB888;
        rga_preproc_.initialize(cfg);
        mat_preproc_width_ = resized_w;
        mat_preproc_height_ = resized_h;
    }
    if (!rga_preproc_.process(source, letterbox_resized_buf_)) return false;
    rga_output_buf_.create(height, width, CV_8UC3);
    rga_output_buf_.setTo(cv::Scalar(114, 114, 114));
    const int left = (width - resized_w) / 2;
    const int top = (height - resized_h) / 2;
    letterbox_resized_buf_.copyTo(
        rga_output_buf_(cv::Rect(left, top, resized_w, resized_h)));
    scale_w = static_cast<float>(resized_w) / source.cols;
    scale_h = static_cast<float>(resized_h) / source.rows;
    pad_x = static_cast<float>(left);
    pad_y = static_cast<float>(top);
    return true;
}

/**
 * @brief 通过 DMA-BUF FD 直接进行推理（硬件直通路径）
 * @param detect_result_group 存储检测结果的结构体
 * @param src_fd 图像数据的 DMA-BUF 文件描述符
 * @param src_width 原始图像宽度
 * @param src_height 原始图像高度
 * @param src_stride 原始图像的内存对齐跨度 (Stride)
 * @param draw_result 是否绘制结果（FD路径通常设为false，在外部按需绘制）
 * @return 0 表示成功，-1 表示失败
 */
int rknn_lite::prepareFromFd(int src_fd, int src_width, int src_height,
                             int src_stride, int src_height_stride,
                             size_t src_buffer_size,
                             pipeline::RgaPixelFormat src_format)
{
    prepared_input_ready_ = false;
    if (rkModel == 0 || !fd_hw_path_ready_ || !zero_copy_input_ready_ ||
        !input_mem_ || src_fd < 0 || src_width <= 0 || src_height <= 0)
    {
        return -1;
    }
    const int source_stride = src_stride > 0 ? src_stride : src_width;
    const int source_height_stride = src_height_stride > 0
        ? src_height_stride : src_height;
    if (source_stride < src_width || source_height_stride < src_height)
    {
        return -1;
    }

    const float scale = std::min(static_cast<float>(width) / src_width,
                                 static_cast<float>(height) / src_height);
    int resized_w = std::min(width, std::max(2,
        static_cast<int>(std::round(src_width * scale)) & ~1));
    int resized_h = std::min(height, std::max(2,
        static_cast<int>(std::round(src_height * scale)) & ~1));
    if (resized_w <= 0 || resized_h <= 0) return -1;
    const int left = (width - resized_w) / 2;
    const int top = (height - resized_h) / 2;
    const bool clear_padding = !fd_padding_valid_ ||
        resized_w != fd_last_resized_w_ || resized_h != fd_last_resized_h_ ||
        left != fd_last_offset_x_ || top != fd_last_offset_y_;

    if (!rga_fd_preproc_.processFdToFdLetterbox(
            src_fd, src_width, src_height, source_stride, source_height_stride,
            src_buffer_size, src_format, input_mem_->fd, width, height,
            input_wstride_, input_hstride_, input_mem_->size,
            resized_w, resized_h, left, top, clear_padding, 114))
    {
        fd_padding_valid_ = false;
        return -1;
    }

    fd_padding_valid_ = true;
    fd_last_resized_w_ = resized_w;
    fd_last_resized_h_ = resized_h;
    fd_last_offset_x_ = left;
    fd_last_offset_y_ = top;

    prepared_scale_w_ = static_cast<float>(resized_w) / src_width;
    prepared_scale_h_ = static_cast<float>(resized_h) / src_height;
    prepared_pad_x_ = static_cast<float>(left);
    prepared_pad_y_ = static_cast<float>(top);
    prepared_input_ready_ = true;
    return 0;
}

int rknn_lite::inferPrepared(detect_result_group_t &detect_result_group,
                             std::mutex* npu_mutex)
{
    if (!prepared_input_ready_ || rkModel == 0 || !input_mem_) return -1;
    prepared_input_ready_ = false;
    const float scale_w = prepared_scale_w_;
    const float scale_h = prepared_scale_h_;
    const float pad_x = prepared_pad_x_;
    const float pad_y = prepared_pad_y_;

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < io_num.n_output; ++i) outputs[i].want_float = 0;

    {
        // Only RKNN runtime/NPU operations hold the global NPU lock. RGA and CPU
        // post-processing remain outside it so other model contexts can progress.
        std::unique_lock<std::mutex> npu_lock;
        if (npu_mutex) npu_lock = std::unique_lock<std::mutex>(*npu_mutex);
        ret = rknn_run(rkModel, nullptr);
        if (ret < 0) return -1;
        ret = rknn_outputs_get(rkModel, io_num.n_output, outputs, nullptr);
        if (ret < 0) return -1;
    }

    const float nms_threshold = nms_threshold_;
    const float box_conf_threshold = confidence_threshold_;
    try {
        if (io_num.n_output >= 6 && io_num.n_output % 3 == 0)
        {
            std::vector<void*> output_buffers(io_num.n_output);
            for (uint32_t i = 0; i < io_num.n_output; ++i) output_buffers[i] = outputs[i].buf;
            ret = post_process_yolov8(output_buffers, output_attrs, io_num.n_output,
                height, width, box_conf_threshold, nms_threshold,
                scale_w, scale_h, pad_x, pad_y,
                &detect_result_group, class_num, label_path_);
        }
        else ret = -1;
    } catch (const std::exception& error) {
        fprintf(stderr, "[rknn_lite] postprocess exception: %s\n", error.what());
        ret = -1;
    } catch (...) {
        fprintf(stderr, "[rknn_lite] postprocess unknown exception\n");
        ret = -1;
    }

    const int postprocess_ret = ret;
    {
        std::unique_lock<std::mutex> npu_lock;
        if (npu_mutex) npu_lock = std::unique_lock<std::mutex>(*npu_mutex);
        if (rknn_outputs_release(rkModel, io_num.n_output, outputs) < 0)
        {
            return -1;
        }
    }
    return postprocess_ret;
}

/**
 * @brief 通过 OpenCV Mat 对象进行推理（传统 CPU/内存路径）
 * @param detect_result_group 存储检测结果的结构体
 * @param draw_result 是否在原始图像上绘制检测框（默认 true）
 * @return 0 表示成功，-1 表示失败
 *
 * 注意：此方法用于 Mat 输入场景（离线图像、非FD红外链路等），
 *       硬件直通 DMA-BUF FD 主链路使用 prepareFromFd()+inferPrepared()。
 */
int rknn_lite::interf(detect_result_group_t &detect_result_group, bool draw_result,
                      std::mutex* npu_mutex)
{
    // 1. 基础有效性检查：确保模型已加载且输入图像不为空
    if (rkModel == 0 || input_attrs == nullptr || output_attrs == nullptr || ori_img.empty())
    {
        return -1;
    }

    cv::Mat source = ori_img; // 局部 header 保持本次调用期间的数据生命周期
    ori_img.release();        // 不让模型成员跨帧长期持有大分辨率 BGR 缓冲

    // 获取输入图像的原始属性
    int img_width = source.cols;  // 原始图像宽度
    int img_height = source.rows; // 原始图像高度
    
    float scale_w = 1.0F, scale_h = 1.0F, pad_x = 0.0F, pad_y = 0.0F;
    // 2. 预处理：优先使用共享letterbox缓冲区，否则走RGA。
    if (!rga_preproc_skip_) {
        if (!preprocessMatLetterbox(
                source, scale_w, scale_h, pad_x, pad_y)) {
            return -1;
        }
    } else {
        rga_preproc_skip_ = false;
        const float scale = std::min(static_cast<float>(width) / img_width,
                                     static_cast<float>(height) / img_height);
        const int resized_w = std::min(width, std::max(2,
            static_cast<int>(std::round(img_width * scale)) & ~1));
        const int resized_h = std::min(height, std::max(2,
            static_cast<int>(std::round(img_height * scale)) & ~1));
        scale_w = static_cast<float>(resized_w) / img_width;
        scale_h = static_cast<float>(resized_h) / img_height;
        pad_x = static_cast<float>((width - resized_w) / 2);
        pad_y = static_cast<float>((height - resized_h) / 2);
    }

    if (rga_output_buf_.empty())
    {
        return -1;
    }

    // set_io_mem 一旦绑定后，两条输入路径必须写入同一块内存。Mat 路径按
    // RKNN 的真实 stride 逐行写入并显式同步 CPU cache。
    if (zero_copy_input_ready_ && !copyMatToBoundInput(rga_output_buf_))
    {
        return -1;
    }

    std::unique_lock<std::mutex> npu_lock;
    if (npu_mutex) npu_lock = std::unique_lock<std::mutex>(*npu_mutex);

    // 未绑定 DMA 输入时保留兼容路径；正常板端路径不再调用 rknn_inputs_set。
    if (!zero_copy_input_ready_)
    {
        inputs[0].buf = (void *)rga_output_buf_.data;
        ret = rknn_inputs_set(rkModel, io_num.n_input, inputs);
        if (ret < 0) return -1;
    }

    // 4. 执行推理：触发 NPU 硬件进行神经网络计算
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < io_num.n_output; i++)
        outputs[i].want_float = 0; // 使用量化输出以获得最高推理性能
    
    ret = rknn_run(rkModel, NULL);
    if (ret < 0)
    {
        return -1;
    }
    
    // 5. 获取输出：从 NPU 寄存器/缓冲区读取推理后的原始张量
    ret = rknn_outputs_get(rkModel, io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        return -1;
    }
    if (npu_lock.owns_lock()) npu_lock.unlock();

    // 6. 后处理参数设置
    const float nms_threshold = nms_threshold_;
    const float box_conf_threshold = confidence_threshold_;

    // 计算坐标映射因子：用于将模型输出的归一化坐标还原到原始图像尺度
    // 7. 量化参数转换：获取模型训练时保留的 Scale 和 ZeroPoint 用于反量化
    // 8. 结果解析：执行后处理算法，将原始张量转换为具体的检测框坐标和类别
    try {
        if (io_num.n_output >= 6 && io_num.n_output % 3 == 0) {
            std::vector<void*> output_buffers(io_num.n_output);
            for (uint32_t i = 0; i < io_num.n_output; ++i) output_buffers[i] = outputs[i].buf;
            ret = post_process_yolov8(output_buffers, output_attrs, io_num.n_output,
                height, width, box_conf_threshold, nms_threshold, scale_w, scale_h, pad_x, pad_y,
                &detect_result_group, class_num, label_path_);
        } else ret = -1;
    } catch (const std::exception& error) {
        fprintf(stderr, "[rknn_lite] postprocess exception: %s\n", error.what());
        ret = -1;
    } catch (...) {
        fprintf(stderr, "[rknn_lite] postprocess unknown exception\n");
        ret = -1;
    }

    const int postprocess_ret = ret;
    {
        std::unique_lock<std::mutex> release_lock;
        if (npu_mutex) release_lock = std::unique_lock<std::mutex>(*npu_mutex);
        if (rknn_outputs_release(rkModel, io_num.n_output, outputs) < 0) return -1;
    }

    // 9. 可选的可视化：释放 RKNN 输出后再绘制，缩短 runtime 资源占用时间。
    if (draw_result)
    {
        char text[256];
        for (int i = 0; i < detect_result_group.count; i++)
        {
            detect_result_t *det_result = &(detect_result_group.results[i]);
            sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
            
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            
            // 针对不同模型 ID (如人员、安全帽等) 使用不同颜色的框
            cv::Scalar color = (id == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            rectangle(source, cv::Point(x1, y1), cv::Point(det_result->box.right, det_result->box.bottom), color, 3);
        }
    }

    return postprocess_ret;
}
/**
 * @brief 从文件中加载数据实现
 * @param fp 文件指针
 * @param ofst 偏移量
 * @param sz 数据大小
 * @return 加载的数据
 */
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

/**
 * @brief 加载模型文件实现
 * @param filename 模型文件名
 * @param model_size 模型大小
 * @return 加载的模型数据
 */
static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

#endif
