/**
 * @file postprocess.h
 * @brief 目标检测后处理头文件
 *
 * 该文件定义了 RKNN NPU 推理后处理相关的数据结构和函数声明。
 *
 * 支持 YOLOv8 NCHW格式DFL边框回归（anchor-free，支持score_sum预过滤）。
 *
 * 核心数据结构：
 * - BOX_RECT：边界框坐标（左上角+右下角，像素坐标系）
 * - detect_result_t：单个检测结果（名称+类别ID+边界框+置信度）
 * - detect_result_group_t：单帧检测结果集合（最多 OBJ_NUMB_MAX_SIZE 个目标）
 *
 * 后处理流程概述：
 * 1. 从量化INT8模型输出中解码边界框坐标和类别置信度
 * 2. 反量化（dequantize）将INT8值恢复为浮点值
 * 3. 按置信度阈值过滤候选框（INT8域快速预过滤 + FP32域精确计算）
 * 4. 按置信度降序排序（快速排序 + 索引映射）
 * 5. 对每个类别单独执行非极大值抑制（per-class NMS）
 * 6. 将模型坐标映射回原始图像坐标（含 letterbox 逆向处理）
 *
 * 量化参数说明：
 * 每个 NPU 输出张量附带 zp（零点）和 scale（比例因子），
 * 用于 INT8 <-> FP32 的仿射转换：f32 = (q - zp) * scale
 */

#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#include <stdint.h>    // int8_t, int32_t 等固定宽度整数类型
#include <string>      // std::string，用于标签路径和类别名称
#include <vector>      // std::vector，用于存储量化参数和中间结果
#include "rknn_api.h"  // Rockchip RKNN API：rknn_tensor_attr（张量属性，含 zp/scale/dims）等类型

#define OBJ_NAME_MAX_SIZE 16  ///< 目标名称字符串的最大长度（含终止符'\0'）
#define OBJ_NUMB_MAX_SIZE 64  ///< 单帧最大检测目标数量上限（NMS 后保留的最终框数）
#define NMS_THRESH        0.50 ///< NMS 默认 IoU 阈值（同一类别内，重叠度超过此值即抑制低置信度框）
#define BOX_THRESH        0.35 ///< 默认置信度阈值（低于此值的检测框被丢弃）

/**
 * @struct BOX_RECT
 * @brief 边界框坐标结构
 *
 * 使用像素坐标系（左上角为原点，x 向右增大，y 向下增大）。
 * 坐标值基于原始图像分辨率（经过 NMS/letterbox逆向/缩放映射后的最终结果）。
 */
typedef struct _BOX_RECT
{
    int left;   ///< 边界框左上角 x 坐标（像素）
    int right;  ///< 边界框右下角 x 坐标（像素）
    int top;    ///< 边界框左上角 y 坐标（像素）
    int bottom; ///< 边界框右下角 y 坐标（像素）
} BOX_RECT;

/**
 * @struct detect_result_t
 * @brief 单个目标的检测结果
 *
 * 包含目标的完整检测信息：类别名称、类别编号、位置和置信度。
 * class_id 与 name 的对应关系由标签文件定义（如 coco_80.txt）。
 */
typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE]; ///< 目标类别名称（如 "person", "car"），以'\0'结尾
    int class_id;                 ///< 模型内部的类别编号（0 ~ class_num-1），用于多模型场景下的精确类别匹配
    BOX_RECT box;                 ///< 目标边界框坐标（原始图像坐标系，像素值）
    float prop;                   ///< 检测置信度（0.0 ~ 1.0），obj_conf * class_conf 或最佳类别分数
} detect_result_t;

/**
 * @struct detect_result_group_t
 * @brief 检测结果组
 *
 * 包含一帧图像中所有经 NMS 后保留的检测目标结果。
 * results 数组按置信度降序排列，通过 count 确定有效元素数量。
 */
typedef struct _detect_result_group_t
{
    int id;                                         ///< 结果组 ID（预留，可用于帧序号标识）
    int count;                                      ///< 有效检测结果数量（0 ~ OBJ_NUMB_MAX_SIZE）
    detect_result_t results[OBJ_NUMB_MAX_SIZE];     ///< 检测结果数组（仅前 count 个有效）
} detect_result_group_t;

/**
 * @brief 设置标签文件路径（在首次推理前调用）
 * @param path 标签文件路径（如 "../model/labels/coco_80.txt"）
 *
 * 允许外部动态配置标签文件路径，替代硬编码的默认路径。
 * 线程安全：内部使用 mutex 保护 g_label_path 的写操作。
 *
 * @note 必须在任何 post_process_yolov8() 调用之前设置，否则使用默认路径。
 * @deprecated 多模型场景请使用 post_process_yolov8() 的 label_path 参数
 *             直接指定路径，该全局设置功能保留用于单模型简化场景。
 */
void post_process_set_label_path(const std::string& path);

/**
 * @brief 强制重置标签缓存
 *
 * 清空内部标签缓存（g_label_cache），使下一个 post_process_yolov8()
 * 调用重新从磁盘加载标签文件。适用于运行中切换模型的场景。
 *
 * 线程安全：内部使用 mutex 保护缓存的写操作。
 */
void post_process_reset_labels();

/**
 * @brief 后处理反初始化函数
 * @param class_num 类别数量（当前未使用，保留用于扩展）
 *
 * 释放后处理模块分配的所有资源。主要在以下场景调用：
 * - 程序退出前清理
 * - 模型切换时重置状态
 *
 * 当前实现：清空标签缓存（g_label_cache）。
 */
void deinitPostProcess(int class_num);

/**
 * @brief Rockchip 优化的 YOLOv8 后处理函数（NCHW DFL 格式，anchor-free）
 *
 * - 输入为 NCHW（通道-高度-宽度）INT8张量
 * - 使用 DFL（Distribution Focal Loss）回归边界框，而非 anchor-based 方法
 * - 每个分支输出 DFL box、class score、score sum 三个张量（共9个张量输出）
 * - 支持 letterbox 填充参数（pad_x/pad_y），正确处理非等比缩放场景
 * - 通过 score_sum 张量实现快速预过滤，跳过明显无目标的网格
 *
 * @param[in] outputs       输出数据指针列表（按 attrs 描述的格式，每个元素指向一个张量）
 * @param[in] attrs         输出张量属性数组（包含维度 dims、量化参数 zp/scale 等）
 * @param[in] output_count  输出张量总数（必须 >= 6 且为 3 的倍数，即每个分支至少2个张量）
 * @param[in] model_in_h    模型输入高度（像素）
 * @param[in] model_in_w    模型输入宽度（像素）
 * @param[in] conf_threshold 置信度阈值
 * @param[in] nms_threshold  NMS IoU 阈值
 * @param[in] scale_w       letterbox 内容宽度 / 原始图像宽度
 * @param[in] scale_h       letterbox 内容高度 / 原始图像高度
 * @param[in] pad_x         letterbox 左侧/右侧填充像素数
 * @param[in] pad_y         letterbox 顶部/底部填充像素数
 * @param[out] group        检测结果组（NMS 后的最终输出）
 * @param[in] class_num     类别数量
 * @param[in] label_path    标签文件路径（可选，空字符串使用全局默认路径）
 * @return 0=成功, -1=参数无效（如 output_count < 6 或 class_num <= 0）
 *
 * YOLOv8 边框解码算法（DFL）：
 * 1. computeYolov8Dfl()：使用 softmax 加权求和，将 dfl_len 个离散 bin 的概率分布
 *    转换为到网格中心的连续距离值（left, top, right, bottom）
 * 2. x1 = (grid_x + 0.5 - distance[0]) * stride
 * 3. y1 = (grid_y + 0.5 - distance[1]) * stride
 * 4. x2 = (grid_x + 0.5 + distance[2]) * stride
 * 5. y2 = (grid_y + 0.5 + distance[3]) * stride
 * 6. 移除 letterbox 填充后，通过 scale_w/scale_h 映射到原始图像坐标
 */
int post_process_yolov8(const std::vector<void*>& outputs,
                        const rknn_tensor_attr* attrs,
                        int output_count,
                        int model_in_h,
                        int model_in_w,
                        float conf_threshold,
                        float nms_threshold,
                        float scale_w,
                        float scale_h,
                        float pad_x,
                        float pad_y,
                        detect_result_group_t* group,
                        int class_num,
                        const std::string& label_path = "");

#endif //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
