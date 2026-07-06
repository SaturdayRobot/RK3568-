/**
 * @file postprocess.cpp
 * @brief 目标检测后处理实现（YOLOv5 anchor-based 和 YOLOv8 DFL anchor-free）
 *
 * 该文件实现了基于 RKNN NPU 推理输出后处理所需的所有功能：
 *
 * --- YOLOv5 后处理流程 (post_process) ---
 * 1. INT8 量化模型输出 -> 反量化为 FP32
 * 2. 每个网格单元 x 3个锚点 -> 解码边界框坐标 (x, y, w, h)
 * 3. 计算 obj_confidence x class_confidence 作为最终置信度
 * 4. 置信度阈值过滤
 * 5. 按置信度降序排序
 * 6. 逐类别执行 NMS（非极大值抑制）
 * 7. 坐标映射回原始图像分辨率
 *
 * --- YOLOv8 后处理流程 (post_process_yolov8) ---
 * 1. 从 NCHW 格式张量中提取 DFL 边界框参数
 * 2. Softmax 加权求和计算到网格中心的 4 个距离值
 * 3. 逐通道扫描类别分数，找出最高置信度类别
 * 4. 置信度过滤 + 排序 + NMS
 * 5. 移除 letterbox 填充后映射到原始图像坐标
 *
 * --- 量化与反量化 ---
 * 使用仿射量化（Affine Quantization）：
 *   quantize:   q = clip(f32 / scale + zp, -128, 127)
 *   dequantize: f32 = (q - zp) * scale
 *
 * --- NPU 推理的关键前提 ---
 * RKNN NPU 针对 INT8 计算进行了深度优化（3 TOPS INT8 vs 0.3 TFLOPS FP16），
 * 因此模型导出时必须做 INT8 量化。后处理负责将量化输出恢复为浮点值。
 */

#include "data_processing/postprocess.h"

#include <cmath>       // exp, fmax, fmin
#include <cstdint>     // int8_t, int32_t
#include <cstdio>      // printf
#include <cstdlib>     // 标准库函数
#include <cstring>     // strncpy, memset

#include <algorithm>   // std::max, std::sort
#include <atomic>      // 原子操作（预留）
#include <fstream>     // std::ifstream（读取标签文件）
#include <mutex>       // std::mutex（线程安全保护标签缓存）
#include <set>         // std::set（提取唯一类别ID）
#include <string>      // std::string, std::to_string
#include <unordered_map> // std::unordered_map（标签缓存容器）
#include <vector>      // std::vector（中间结果存储）

#define LABEL_NALE_TXT_PATH "../model/labels/coco_trimmed.txt"  ///< 默认COCO标签文件路径（裁剪版）

// === 标签缓存：线程安全的多模型标签管理 ===
// g_label_cache: 按路径缓存，不同模型使用不同路径时互不覆盖
// g_labels_mutex: 保护缓存读写的互斥锁
// g_label_path: 全局默认标签路径（可通过 post_process_set_label_path 修改）
static std::unordered_map<std::string, std::vector<std::string>> g_label_cache;
static std::mutex g_labels_mutex;
static std::string g_label_path = LABEL_NALE_TXT_PATH;

// === 锚点数组定义 ===
// YOLOv5 使用 3 个输出分支，每个分支 3 个锚点（共 9 个锚点）
// 锚点格式：[w0, h0, w1, h1, w2, h2]，单位为模型输入空间像素
// 这些锚点值来源于训练数据集的 K-Means 聚类结果，针对 COCO 数据集优化
// 步长 8 的锚点用于检测小目标（在 80x80 特征图上，感受野小）
const int anchor0[6] = {10, 13, 16, 30, 33, 23};
// 步长 16 的锚点用于检测中目标（在 40x40 特征图上，感受野中等）
const int anchor1[6] = {30, 61, 62, 45, 59, 119};
// 步长 32 的锚点用于检测大目标（在 20x20 特征图上，感受野大）
const int anchor2[6] = {116, 90, 156, 198, 373, 326};

/**
 * @brief 将浮点值限制在指定范围内
 * @param val 输入值
 * @param min 最小值
 * @param max 最大值
 * @return 限制后的值：min <= return <= max
 *
 * 用于坐标裁剪，防止检测框越界到图像外。
 * 注意：返回值是 int 类型，实际有 float->int 的隐式截断。
 */
inline static int clamp(float val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;  // 三目运算符实现的范围钳位
}

void post_process_set_label_path(const std::string& path) {
    std::lock_guard<std::mutex> lock(g_labels_mutex);  // 加锁保护写操作
    g_label_path = path;  // 更新全局默认标签路径
}

void post_process_reset_labels() {
    std::lock_guard<std::mutex> lock(g_labels_mutex);  // 加锁保护写操作
    g_label_cache.clear();  // 清空所有缓存的标签数据
}

/**
 * @brief 线程安全地加载标签文件（按路径缓存）
 * @param class_num    需要的类别数量
 * @param override_path 覆盖路径（空字符串使用 g_label_path）
 * @return 标签名称的副本（调用方线程安全持有）
 *
 * 缓存策略：
 * - 快速路径：缓存命中时直接返回副本（O(1)，无磁盘IO）
 * - 慢速路径：首次加载时从文件读取，存入缓存
 * - 文件不存在时自动生成 "class_0", "class_1", ... 作为回退标签
 * - 文件行数不足时用 "unknown" 补齐
 *
 * 线程安全设计：
 * - 使用双重检查锁定（Double-Checked Locking）模式
 * - 先在不持锁的情况下检查缓存（读路径），未命中时才加锁写入
 * - 磁盘IO（慢速路径）在锁外执行，避免长时间持锁阻塞其他线程
 */
static std::vector<std::string> loadLabelsOnce(int class_num, const std::string& override_path = "") {
    // 确定使用的标签路径：显式传入 > 全局默认
    const std::string path = override_path.empty() ? g_label_path : override_path;

    // 快速路径：检查缓存（持锁时间短，仅做查找和拷贝）
    {
        std::lock_guard<std::mutex> lock(g_labels_mutex);
        auto it = g_label_cache.find(path);  // O(1) 哈希查找
        if (it != g_label_cache.end()) {
            return it->second;  // 返回缓存副本（vector 拷贝构造）
        }
    }  // 锁在此处释放，不阻塞后续磁盘IO

    // 慢速路径：从磁盘加载标签文件（IO 操作在锁外执行）
    std::vector<std::string> labels;
    std::ifstream file(path);  // 打开标签文件（文本模式）
    if (!file.is_open()) {
        // 文件不存在：生成回退标签 class_0, class_1, ...
        labels.reserve(static_cast<size_t>(class_num));  // 预分配内存，避免多次扩容
        for (int i = 0; i < class_num; ++i) {
            labels.push_back("class_" + std::to_string(i));
        }
        printf("loadLabelsOnce: %s not found, using fallback labels\n", path.c_str());
    } else {
        // 文件存在：逐行读取类别名称
        labels.reserve(static_cast<size_t>(class_num));
        std::string line;
        while (std::getline(file, line) && static_cast<int>(labels.size()) < class_num) {
            // 去除行尾的 \r, \n, 空格字符（兼容 Windows/Linux 换行格式）
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
                line.pop_back();
            }
            labels.push_back(std::move(line));  // 移动语义，避免字符串拷贝
        }
        // 如果文件行数不足 class_num，用 "unknown" 补齐
        while (static_cast<int>(labels.size()) < class_num) {
            labels.push_back("unknown");
        }
        printf("loadLabelsOnce: loaded %zu labels from %s\n", labels.size(), path.c_str());
    }

    // 存入缓存（再次加锁，写入操作）
    {
        std::lock_guard<std::mutex> lock(g_labels_mutex);
        g_label_cache[path] = labels;  // 存入缓存（拷贝）
    }
    return labels;  // 返回副本
}

/**
 * @brief 计算两个边界框的交并比（IoU, Intersection over Union）
 *
 * 算法：
 *   intersection_width  = max(0, min(xmax0, xmax1) - max(xmin0, xmin1) + 1)
 *   intersection_height = max(0, min(ymax0, ymax1) - max(ymin0, ymin1) + 1)
 *   intersection = w * h
 *   union = area0 + area1 - intersection
 *   IoU = intersection / union
 *
 * @param xmin0,ymin0 第一个框的左上角坐标
 * @param xmax0,ymax0 第一个框的右下角坐标
 * @param xmin1,ymin1 第二个框的左上角坐标
 * @param xmax1,ymax1 第二个框的右下角坐标
 * @return IoU 值（0.0 ~ 1.0），并集为 0 时返回 0.0
 *
 * @note +1.0 偏移量用于包含边界像素（像素坐标含边界的传统做法）。
 *       这是 NMS 特定的 IoU 计算，与 classification/instance-seg 中不含 +1.0
 *       连续坐标空间的标准 IoU 有细微差异。
 */
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
                               float xmin1, float ymin1, float xmax1, float ymax1)
{
    // 交集宽度：取两框右边界最小值 - 左边界最大值 + 1（含边界像素）
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    // 交集高度：取两框下边界最小值 - 上边界最大值 + 1（含边界像素）
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;  // 交集面积

    // 并集面积 = area0 + area1 - intersection
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) +
              (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) -
              i;

    return u <= 0.f ? 0.f : (i / u);  // 防止除零：并集面积为0时返回0
}

/**
 * @brief 非极大值抑制（NMS, Non-Maximum Suppression）
 *
 * NMS 是目标检测后处理的核心步骤，用于消除对同一目标的重复检测。
 *
 * 算法：
 * 1. 遍历已按置信度降序排列的检测框
 * 2. 对每个框，与后续同类别且 IoU > threshold 的框比较
 * 3. 将 IoU 超过阈值的低置信度框标记为无效（order[j] = -1）
 *
 * 使用逐类别 NMS（per-class NMS）而非类别无关 NMS（class-agnostic NMS），
 * 确保不同类别的重叠目标（如"人"和"自行车"重叠）不被误抑制。
 *
 * @param validCount       有效候选框总数
 * @param outputLocations  边界框坐标数组（每框4个值：(x, y) 左上角 + (w, h) 宽高）
 * @param classIds         类别ID数组
 * @param order            排序后的索引数组（按置信度从高到低），被原地修改以标记抑制的框
 * @param filterId         当前只处理此类别的框
 * @param threshold        IoU 阈值：超过此值即抑制较低置信度的框
 * @return 0=成功
 */
static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds,
               std::vector<int> &order, int filterId, float threshold)
{
    // 遍历所有候选框（按置信度降序）
    for (int i = 0; i < validCount; ++i)
    {
        // 跳过已标记为无效的框，跳过非当前处理类别的框
        if (order[i] == -1 || classIds[order[i]] != filterId)
        {
            continue;
        }

        int n = order[i];  // 当前保留框的原始索引（置信度较高者）

        // 遍历当前框之后的所有框（置信度较低或相等的框）
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];  // 待比较框的原始索引

            // 跳过已标记无效或不同类别的框
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }

            // 计算第一个框的边界坐标：将 (x, y, w, h) 转换为 (xmin, ymin, xmax, ymax)
            float xmin0 = outputLocations[n * 4 + 0];                               // 左上角 x
            float ymin0 = outputLocations[n * 4 + 1];                               // 左上角 y
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];  // 右下角 x = x + w
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];  // 右下角 y = y + h

            // 计算第二个框的边界坐标
            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            // 计算两个框的 IoU
            const float iou = CalculateOverlap(
                xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            // IoU 超过阈值：抑制该框（标记为 -1，后续处理时跳过）
            if (iou > threshold)
            {
                order[j] = -1;  // 标记为已被抑制
            }
        }
    }
    return 0;
}

/**
 * @brief 快速排序（降序，原地排序，同时维护索引映射）
 *
 * 对 input 数组按降序排序，同时同步更新 indices 数组保持对应关系。
 * 采用快速排序算法（Lomuto 分区方案变体），时间复杂度 O(n log n)，
 * 空间复杂度 O(log n)（递归栈）。
 *
 * @param input   待排序的值数组（置信度等），排序后原地改变
 * @param left    排序范围的左边界索引（含）
 * @param right   排序范围的右边界索引（含）
 * @param indices 索引映射数组，与 input 同步更新
 * @return 基准值最终位置索引
 */
static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right,
                                      std::vector<int> &indices)
{
    float key;        // 基准值（pivot value）
    int key_index;    // 基准值对应的原始索引
    int low = left;   // 左扫描指针
    int high = right; // 右扫描指针

    if (left < right)
    {
        // 选取最左元素作为基准值
        key_index = indices[left];
        key = input[left];

        // 双向扫描：从两端交替向中间移动
        while (low < high)
        {
            // 从右向左找第一个大于基准值的元素（降序排列：大值在左）
            while (low < high && input[high] <= key)
            {
                high--;
            }
            // 将该元素移到左边空位
            input[low] = input[high];
            indices[low] = indices[high];

            // 从左向右找第一个小于基准值的元素（降序排列：小值在右）
            while (low < high && input[low] >= key)
            {
                low++;
            }
            // 将该元素移到右边空位
            input[high] = input[low];
            indices[high] = indices[low];
        }

        // 基准值归位
        input[low] = key;
        indices[low] = key_index;

        // 递归排序左半部分
        quick_sort_indice_inverse(input, left, low - 1, indices);
        // 递归排序右半部分
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;  // 返回基准值最终位置
}

/**
 * @brief 将浮点值裁剪到 [min, max] 范围
 * @param val 输入值
 * @param min 最小值
 * @param max 最大值
 * @return 裁剪后的值（float->int32_t 截断）
 */
inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;  // 隐式 float->int32_t 截断
}

/**
 * @brief FP32 仿射量化为 INT8
 *
 * 公式：q = clip(round(f32 / scale + zp), -128, 127)
 *
 * 用于将浮点阈值转换为 INT8 量化值，使得置信度阈值比较可在 INT8 域直接进行，
 * 避免每个候选框都做反量化操作，显著提升后处理速度。
 *
 * @param f32   输入浮点值
 * @param zp    零点偏移（Zero Point），INT32 类型以支持非对称量化
 * @param scale 量化比例因子
 * @return 量化后的 INT8 值
 */
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    // 应用量化公式：浮点值缩放到目标范围后再加零点
    float dst_val = (f32 / scale) + zp;
    // 裁剪到 INT8 有效范围 [-128, 127] 并转换为 int8_t
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

/**
 * @brief INT8 仿射反量化为 FP32
 *
 * 公式：f32 = (q - zp) * scale
 *
 * 将 NPU 输出的 INT8 张量值恢复为原始浮点值，
 * 用于后续的边界框坐标计算和置信度计算。
 *
 * @param qnt   量化后的 INT8 值
 * @param zp    零点偏移（Zero Point）
 * @param scale 量化比例因子
 * @return 反量化后的 FP32 值
 */
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    // 应用反量化公式：先减去零点，再乘以比例因子
    return ((float)qnt - (float)zp) * scale;
}

/**
 * @brief 处理单个输出层：解码边界框并过滤低置信度候选框
 *
 * YOLOv5 的每层输出为 NCHW 格式（通道-高度-宽度），
 * 每个网格单元输出 (5 + class_num) * 3 个值（3个锚点）。
 * 通道布局：[anchor0 {tx, ty, tw, th, obj, class0..classN}, anchor1{...}, anchor2{...}]
 *
 * 边界框解码公式（YOLOv5）：
 *   box_x = (sigmoid(tx) * 2 - 0.5 + grid_x) * stride
 *   box_y = (sigmoid(ty) * 2 - 0.5 + grid_y) * stride
 *   box_w = (sigmoid(tw) * 2)^2 * anchor_w
 *   box_h = (sigmoid(th) * 2)^2 * anchor_h
 *
 * 重要说明：YOLOv5 在 ONNX 导出时已将 sigmoid 融合到模型中
 * （pytorch 的 export 流程中，Detect 层的 forward 返回已过 sigmoid 的值）。
 * 因此代码中直接使用反量化后的值，无需再显式计算 sigmoid。
 *
 * @param[in]  input      INT8 量化模型输出数据（NCHW 格式，已展平为一维）
 * @param[in]  anchor     当前层的锚点数组（3对 w,h 值，单位：模型输入空间像素）
 * @param[in]  grid_h     特征图高度（网格行数）
 * @param[in]  grid_w     特征图宽度（网格列数）
 * @param[in]  height     模型输入高度
 * @param[in]  width      模型输入宽度
 * @param[in]  stride     当前层的步长（下采样倍率，= model_h / grid_h）
 * @param[out] boxes      边界框坐标输出（逐个追加 x, y, w, h）
 * @param[out] objProbs   目标置信度输出（obj_conf * class_conf）
 * @param[out] classId    类别ID输出
 * @param[in]  threshold  置信度阈值（浮点值，将为INT8比较而量化）
 * @param[in]  zp         量化零点偏移
 * @param[in]  scale      量化比例因子
 * @param[in]  class_num  类别总数
 * @return 本层输出的有效候选框数量
 */
static int process(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width,
                   int stride, std::vector<float> &boxes, std::vector<float> &objProbs,
                   std::vector<int> &classId, float threshold, int32_t zp, float scale,
                   int class_num)
{
    int validCount = 0;               // 有效检测框累计计数
    int grid_len = grid_h * grid_w;   // 网格单元总数（空间维度展平后的长度）

    // 将浮点阈值转换为 INT8 量化值，避免每个候选框都做反量化比较
    // 这是一个关键的性能优化：在 INT8 域做阈值比较，仅在通过后才反量化计算精确坐标
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

    // 遍历 3 个锚点（每个网格预测 3 个边界框，对应不同的宽高比）
    for (int a = 0; a < 3; a++)
    {
        // 遍历每个网格行
        for (int i = 0; i < grid_h; i++)
        {
            // 遍历每个网格列
            for (int j = 0; j < grid_w; j++)
            {
                // 计算目标置信度（obj_conf）在 NCHW 数据中的位置
                // NCHW 布局：通道维度上依次排列 3 组，每组 (5 + class_num) 个通道
                // obj_conf 是每组中索引为 4 的通道（前4个是 tx, ty, tw, th）
                int8_t box_confidence = input[((5 + class_num) * a + 4) * grid_len + i * grid_w + j];

                // 快速过滤：目标置信度低于阈值直接跳过（INT8 域比较，无需反量化）
                if (box_confidence >= thres_i8)
                {
                    // 计算当前锚点和网格单元在展平数据中的起始偏移
                    // offset 指向 tx（当前锚点、当前网格的边界框参数起始位置）
                    int offset = ((5 + class_num) * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input + offset;  // 定位到当前预测的起始地址

                    // --- YOLOv5 边界框解码 ---
                    // 注意：tx, ty, tw, th 在 NCHW 中按通道维度连续排列，间隔为 grid_len
                    // tx: in_ptr[0], ty: in_ptr[grid_len], tw: in_ptr[2*grid_len], th: in_ptr[3*grid_len]

                    // tx, ty：预测的边界框中心偏移（模型输出已过 sigmoid，无需再计算）
                    // 公式：box_center = sigmoid(pred) * 2 - 0.5
                    // YOLOv5 解码策略：将中心点限制在当前网格附近（[-0.5, 1.5] 范围），避免偏移过大
                    float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;

                    // tw, th：预测的宽高缩放因子（模型输出已过 sigmoid，无需再计算）
                    // 公式：box_size = (sigmoid(pred) * 2)^2 * anchor_size
                    float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;

                    // 从网格坐标转换为图像坐标（模型输入空间）
                    // 中心点坐标 = (相对偏移 + 网格索引) * 步长
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    // 宽高 = 缩放因子^2 * 锚点尺寸
                    box_w = box_w * box_w * (float)anchor[a * 2];      // a*2 = 锚点宽度索引
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];  // a*2+1 = 锚点高度索引

                    // 将中心坐标转换为左上角坐标（统一使用 (x, y, w, h) 格式）
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    // 在当前预测的 class_num 个类别分数中找出最大值及对应类别
                    int8_t maxClassProbs = in_ptr[5 * grid_len];  // 第一个类别分数（class_0，在通道5）
                    int maxClassId = 0;
                    for (int k = 1; k < class_num; ++k)
                    {
                        int8_t prob = in_ptr[(5 + k) * grid_len];  // 第 k 个类别分数
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }

                    // 类别置信度阈值过滤（INT8 域比较）
                    if (maxClassProbs > thres_i8)
                    {
                        // 计算最终置信度 = (反量化后的)类别置信度 * (反量化后的)目标置信度
                        // 这里必须反量化为 FP32，因为在阈值比较后需要精确的置信度值供后续 NMS 排序使用
                        objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) *
                                           (deqnt_affine_to_f32(box_confidence, zp, scale)));
                        classId.push_back(maxClassId);  // 保存类别ID
                        validCount++;                    // 有效框计数+1
                        // 添加边界框坐标：左上角 x, y, 宽度 w, 高度 h
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;  // 返回本层有效候选框总数
}

/**
 * @brief 目标检测后处理主函数（YOLOv5 风格，anchor-based，3输出分支）
 *
 * 完整后处理管道：
 *
 *   INT8输入(x3层) -> process()解码+过滤 -> 合并所有候选框
 *     -> 按置信度降序排序 -> 逐类别NMS -> 坐标映射 -> 填充group
 *
 * 坐标映射说明：
 * - 模型输入经 letterbox resize 后可能不等于原始图像尺寸
 * - scale_w = 原始图像宽度 / 模型输入宽度（即映射比例因子）
 * - scale_h = 原始图像高度 / 模型输入高度
 * - 模型坐标 / scale = 原始图像坐标（注意：letterbox 场景需额外处理 pad）
 *
 * @return 0=成功（即使无检测结果也返回0），负值=失败
 */
int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group, int class_num, const std::string& label_path)
{
    // 加载标签（线程安全，按路径缓存）
    std::vector<std::string> labels = loadLabelsOnce(class_num, label_path);

    // === 中间结果容器 ===
    std::vector<float> filterBoxes;   // 所有候选框的坐标（每框4个float：x, y, w, h）
    filterBoxes.reserve(1024);        // 预分配减少热路径 vector 扩容开销
    std::vector<float> objProbs;      // 所有候选框的最终置信度（obj_conf * class_conf）
    objProbs.reserve(256);
    std::vector<int> classId;         // 所有候选框的类别ID
    classId.reserve(256);

    // === 处理步长 8 的输出层（80x80 特征图，检测小目标）===
    // 步长越小，特征图分辨率越高，负责检测越小的目标
    int stride0 = 8;                                            // 步长：8倍下采样（相对于输入）
    int grid_h0 = model_in_h / stride0;                         // 特征图高度 = 模型高度/步长
    int grid_w0 = model_in_w / stride0;                         // 特征图宽度 = 模型宽度/步长
    int validCount0 = 0;
    validCount0 = process(input0, (int *)anchor0, grid_h0, grid_w0, model_in_h, model_in_w,
                          stride0, filterBoxes, objProbs, classId,
                          conf_threshold, qnt_zps[0], qnt_scales[0], class_num);

    // === 处理步长 16 的输出层（40x40 特征图，检测中目标）===
    int stride1 = 16;
    int grid_h1 = model_in_h / stride1;
    int grid_w1 = model_in_w / stride1;
    int validCount1 = 0;
    validCount1 = process(input1, (int *)anchor1, grid_h1, grid_w1, model_in_h, model_in_w,
                          stride1, filterBoxes, objProbs, classId,
                          conf_threshold, qnt_zps[1], qnt_scales[1], class_num);

    // === 处理步长 32 的输出层（20x20 特征图，检测大目标）===
    int stride2 = 32;
    int grid_h2 = model_in_h / stride2;
    int grid_w2 = model_in_w / stride2;
    int validCount2 = 0;
    validCount2 = process(input2, (int *)anchor2, grid_h2, grid_w2, model_in_h, model_in_w,
                          stride2, filterBoxes, objProbs, classId,
                          conf_threshold, qnt_zps[2], qnt_scales[2], class_num);

    // 汇总三层有效框总数
    int validCount = validCount0 + validCount1 + validCount2;

    // 无目标检测：直接返回成功（空结果也是合法的输出）
    if (validCount <= 0)
    {
        return 0;
    }

    // === 创建索引数组用于排序 ===
    std::vector<int> indexArray(validCount);
    for (int i = 0; i < validCount; ++i)
    {
        indexArray[i] = i;  // 初始化索引为 0, 1, 2, ...
    }

    // === 按置信度降序排序 ===
    // quick_sort_indice_inverse: 对 objProbs 降序排列，同步更新 indexArray
    // 排序后的 indexArray[0] 指向置信度最高的框，indexArray[-1] 指向最低的框
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    // === 提取所有唯一的类别ID ===
    // 使用 std::set 自动去重和排序，用于后续逐类别 NMS
    std::set<int> class_set(std::begin(classId), std::end(classId));

    // === 逐类别执行 NMS ===
    // 关键设计选择：使用 per-class NMS 而非 class-agnostic NMS
    // 原因：不同类别的目标即使高度重叠（如"人"骑"自行车"），也不应互相抑制。
    // 如果使用 class-agnostic NMS，骑自行车的人可能被错误地抑制掉。
    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    // === 构建最终检测结果组 ===
    int last_count = 0;   // 最终有效结果计数（NMS 后保留的框数）
    group->count = 0;     // 初始化结果数量

    for (int i = 0; i < validCount; ++i)
    {
        // 跳过已被 NMS 抑制的框（order[j] = -1），以及达到最大数量上限
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }

        int n = indexArray[i];  // 获取排序后的原始索引（指向原始框数据）

        // 提取边界框坐标
        float x1 = filterBoxes[n * 4 + 0];      // 左上角 x（模型输入空间）
        float y1 = filterBoxes[n * 4 + 1];      // 左上角 y（模型输入空间）
        float x2 = x1 + filterBoxes[n * 4 + 2]; // 右下角 x = x + w
        float y2 = y1 + filterBoxes[n * 4 + 3]; // 右下角 y = y + h

        int id = classId[n];           // 类别ID
        float obj_conf = objProbs[i];  // 置信度（排序后可能不等于原始顺序的值，但近似）

        // 坐标映射：模型输入空间 -> 原始图像空间
        // 1. clamp: 裁剪到模型输入范围内（防止越界坐标）
        // 2. /scale: 将模型坐标缩放到原始图像分辨率
        //    scale_w = orig_width / model_width, 所以 model_coord / scale_w = orig_coord
        group->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / scale_w);
        group->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / scale_h);
        group->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / scale_w);
        group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);

        group->results[last_count].class_id = id;

        // 设置类别名称：优先使用标签文件中的名称，回退到 "class_N" 格式
        const std::string fallback_name = "class_" + std::to_string(id);
        const std::string& result_name =
            (id >= 0 && static_cast<size_t>(id) < labels.size()) ? labels[id] : fallback_name;
        // strncpy：安全字符串拷贝，保证不超过缓冲区大小（OBJ_NAME_MAX_SIZE）
        strncpy(group->results[last_count].name, result_name.c_str(), OBJ_NAME_MAX_SIZE - 1);
        group->results[last_count].name[OBJ_NAME_MAX_SIZE - 1] = '\0';  // 强制终止符

        group->results[last_count].prop = obj_conf;  // 置信度

        last_count++;  // 有效结果计数+1
    }

    group->count = last_count;  // 设置最终检测结果总数

    return 0;
}

/**
 * @brief 后处理模块反初始化
 *
 * 清空标签缓存，释放持有的内存。当前实现仅清理缓存，
 * class_num 参数保留用于未来扩展（如释放每个模型独立分配的资源）。
 */
void deinitPostProcess(int class_num)
{
    (void)class_num;  // 显式标记当前未使用，消除编译警告
    std::lock_guard<std::mutex> lock(g_labels_mutex);
    g_label_cache.clear();  // 释放所有缓存的标签数据
}

namespace {
/**
 * @brief YOLOv8 DFL（Distribution Focal Loss）边界框解码
 *
 * DFL 将边界框的每条边（left, top, right, bottom）建模为 dfl_len 个离散概率值，
 * 通过对概率分布进行 softmax 加权求和来计算精确的边界距离。
 *
 * 算法步骤（对每条边）：
 * 1. 找到 dfl_len 个 logit 中的最大值（用于 softmax 数值稳定）
 * 2. 计算 softmax: p_i = exp(logit_i - max) / sum(exp(logit_j - max))
 * 3. 加权求和: distance = sum(p_i * i), i = 0,1,...,dfl_len-1
 *
 * 数值稳定技巧（减去最大值）：
 * 直接计算 exp(large_logit) 会导致浮点溢出（exp(88)=~1.6e38，超越 FP32 范围）。
 * 减去最大值后所有指数值 <= 1，安全可控。
 *
 * 该解码方式相比 anchor-based 回归的优势：
 * - 无需预设锚点（anchor-free），适应任意宽高比的目标
 * - 对边界框定位更精确（通过分布建模而非单点回归）
 * - 数值更稳定（softmax 输出在 0~1 之间，且使用 log-sum-exp 技巧）
 *
 * @param[in]  box_data     NCHW 格式的边框回归输出（已量化INT8）
 * @param[in]  grid_stride  网格步长 = grid_h * grid_w（展平空间维度的总元素数）
 * @param[in]  position     当前网格单元在展平空间中的位置索引
 * @param[in]  zp           量化零点偏移
 * @param[in]  scale        量化比例因子
 * @param[in]  dfl_len      DFL 回归向量的长度（每个边离散化后的 bin 数量），通常为16
 * @param[out] box          输出的4个距离值：[left, top, right, bottom]
 */
void computeYolov8Dfl(const int8_t* box_data, int grid_stride, int position,
                      int32_t zp, float scale, int dfl_len, float box[4]) {
    for (int side = 0; side < 4; ++side)  // 遍历4条边：left(0), top(1), right(2), bottom(3)
    {
        // 第一步：找到最大值用于 softmax 数值稳定
        // 不减去最大值直接 exp 可能导致数值溢出：
        // 例如 exp(90) ≈ 1.2e39 超出 FP32 最大值约 3.4e38
        float maximum = -INFINITY;
        for (int i = 0; i < dfl_len; ++i) {
            const int c = side * dfl_len + i;  // NCHW 通道索引：
                                                // channels = [left_bins(0..15), top_bins(16..31),
                                                //             right_bins(32..47), bottom_bins(48..63)]
            // 反量化 INT8 -> FP32 后取最大值
            maximum = std::max(maximum,
                deqnt_affine_to_f32(box_data[c * grid_stride + position], zp, scale));
        }

        // 第二步：计算 softmax 加权和
        // distance = sum(exp(logit_i - max) * i) / sum(exp(logit_i - max))
        // 这是期望值公式 E[X] = sum(p_i * i)，其中 p_i = softmax(logit_i)
        float sum = 0.0F;     // softmax 分母（所有权重之和）
        float weighted = 0.0F; // softmax 分子（权重 * 索引 之和）
        for (int i = 0; i < dfl_len; ++i) {
            const int c = side * dfl_len + i;
            // exp(logit - max)：softmax 的分子部分（未经归一化）
            const float value = std::exp(
                deqnt_affine_to_f32(box_data[c * grid_stride + position], zp, scale) - maximum);
            sum += value;        // 累加所有权重（用于归一化）
            weighted += value * i; // 累加权重 * bin索引
        }
        // 加权平均值即为边界距离（以特征图像素为单位）
        // sum > 0 的检查避免了除零：理论上最小值 max 对应 logit，其 exp(0)=1，sum≥1
        box[side] = sum > 0.0F ? weighted / sum : 0.0F;
    }
}
}  // namespace

/**
 * @brief YOLOv8 目标检测后处理（Rockchip 优化版，NCHW DFL 格式，anchor-free）
 *
 * YOLOv8 输出格式：
 * 每个分支输出 2~3 个张量（取决于 outputs_per_branch）：
 * - DFL Box 张量：NCHW 格式，通道数 = dfl_len * 4（4条边各 dfl_len 个 bin）
 * - Class Score 张量：NCHW 格式，通道数 = class_num
 * - Score Sum 张量（可选）：用于快速预过滤，值为各类别分数之和
 *
 * 后处理管道：
 *   NCHW张量(x6~9) -> DFL解码距离 -> 网格坐标计算x1,y1,x2,y2
 *     -> 类别分数扫描 -> 置信度过滤 -> 排序 -> 逐类别NMS
 *     -> letterbox逆向 -> 坐标映射 -> 填充group
 *
 * @return 0=成功, -1=参数无效, 正数=其他错误
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
                        const std::string& label_path) {
    // 参数有效性校验：output_count 必须 >= 6（至少2张量×3分支）且为3的倍数
    if (!attrs || !group || output_count < 6 || output_count % 3 != 0 ||
        static_cast<int>(outputs.size()) < output_count || class_num <= 0) return -1;

    // 加载标签
    std::vector<std::string> labels = loadLabelsOnce(class_num, label_path);

    // === 中间结果容器 ===
    std::vector<float> boxes;        // 边界框坐标（每框4个float：x, y, w, h）
    std::vector<float> probabilities; // 置信度（最佳类别的分数）
    std::vector<int> classes;         // 类别ID

    // YOLOv8 输出分为 branches=3 个分支（对应3个下采样步长）
    const int outputs_per_branch = output_count / 3;  // 每个分支的张量数量（2或3）

    for (int branch = 0; branch < 3; ++branch)
    {
        // 计算当前分支各张量在 outputs/attrs 中的索引
        const int box_index = branch * outputs_per_branch;        // DFL 边框回归张量索引
        const int score_index = box_index + 1;                     // 类别分数张量索引
        const int sum_index = outputs_per_branch == 3 ? box_index + 2 : -1;  // 分数和张量索引（可选）

        const auto& box_attr = attrs[box_index];      // 边框张量属性（含维度、量化参数）
        const auto& score_attr = attrs[score_index];  // 分数张量属性
        // 检查数据有效性和维度（NCHW 至少需要4维）
        if (!outputs[box_index] || !outputs[score_index] || box_attr.n_dims < 4) return -1;

        // 解析张量维度（NCHW 格式：N=1, C=channels, H=height, W=width）
        const int grid_h = static_cast<int>(box_attr.dims[2]);  // H 维度：特征图高度
        const int grid_w = static_cast<int>(box_attr.dims[3]);  // W 维度：特征图宽度
        const int box_channels = static_cast<int>(box_attr.dims[1]); // C 维度：边框通道数 = dfl_len * 4
        const int dfl_len = box_channels / 4;  // 每条边的 DFL bin 数量（标准 YOLOv8 为 16）

        if (grid_h <= 0 || grid_w <= 0 || dfl_len <= 0) return -1;

        const int stride = model_in_h / grid_h;  // 当前分支的下采样步长（=8, 16, 32）

        // 获取各张量的数据指针
        const auto* box_data = static_cast<const int8_t*>(outputs[box_index]);    // 边框回归
        const auto* score_data = static_cast<const int8_t*>(outputs[score_index]); // 类别分数
        const auto* sum_data = sum_index >= 0 ? static_cast<const int8_t*>(outputs[sum_index]) : nullptr;

        const int grid_stride = grid_h * grid_w;  // 空间维度的总元素数（用于 NCHW 索引计算）

        // === 遍历特征图的每个网格单元 ===
        for (int y = 0; y < grid_h; ++y)
        {
            for (int x = 0; x < grid_w; ++x)
            {
                const int position = y * grid_w + x;  // 当前网格在展平空间中的索引

                // 快速预过滤：通过 score_sum 跳过明显无目标的网格
                // score_sum 是所有 class_num 个类别分数的总和，由 RKNN 模型导出时可选生成
                // 如果总分都低于阈值，则任何单个类别必然低于阈值，可安全跳过
                if (sum_data)
                {
                    const float score_sum = deqnt_affine_to_f32(
                        sum_data[position], attrs[sum_index].zp, attrs[sum_index].scale);
                    if (score_sum < conf_threshold) continue;  // 总分低于阈值，跳过此网格
                }

                // === 扫描类别分数，找出最高置信度的类别 ===
                int best_class = -1;
                float best_score = conf_threshold;  // 最低合格分数即为阈值

                // NCHW 格式索引：score_data[c * grid_stride + position]
                // 遍历所有 class_num 个通道中的类别分数
                for (int c = 0; c < class_num; ++c)
                {
                    const float score = deqnt_affine_to_f32(
                        score_data[c * grid_stride + position], score_attr.zp, score_attr.scale);
                    if (score > best_score)
                    {
                        best_score = score;
                        best_class = c;
                    }
                }
                if (best_class < 0) continue;  // 没有类别超过阈值

                // === DFL 解码：计算边界框到网格中心的距离 ===
                // distance[0]=left, distance[1]=top, distance[2]=right, distance[3]=bottom
                // 距离值以特征图像素为单位，乘以 stride 后映射到模型输入空间
                float distance[4]{};
                computeYolov8Dfl(box_data, grid_stride, position,
                                 box_attr.zp, box_attr.scale, dfl_len, distance);

                // === 计算边界框的绝对坐标（模型输入空间）===
                // YOLOv8 公式（网格中心点 + 0.5 偏移因为网格单元以像素中心为原点）：
                //   x1 = (grid_x + 0.5 - left)   * stride
                //   y1 = (grid_y + 0.5 - top)    * stride
                //   x2 = (grid_x + 0.5 + right)  * stride
                //   y2 = (grid_y + 0.5 + bottom) * stride
                const float x1 = (x + 0.5F - distance[0]) * stride;
                const float y1 = (y + 0.5F - distance[1]) * stride;
                const float x2 = (x + 0.5F + distance[2]) * stride;
                const float y2 = (y + 0.5F + distance[3]) * stride;

                // 存储为 (x, y, w, h) 格式（与标准 NMS 函数接口兼容）
                boxes.insert(boxes.end(), {x1, y1, x2 - x1, y2 - y1});
                probabilities.push_back(best_score);
                classes.push_back(best_class);
            }
        }
    }

    // === 排序与 NMS（与 YOLOv5 后处理共享相同逻辑）===
    const int valid_count = static_cast<int>(probabilities.size());
    group->count = 0;
    if (valid_count == 0) return 0;  // 无候选框，直接返回

    // 创建并初始化排序索引数组
    std::vector<int> order(static_cast<size_t>(valid_count));
    for (int i = 0; i < valid_count; ++i) order[i] = i;

    // 按置信度降序排序
    quick_sort_indice_inverse(probabilities, 0, valid_count - 1, order);

    // 提取所有唯一类别ID
    std::set<int> class_set(classes.begin(), classes.end());

    // 逐类别执行 NMS（per-class NMS，理由同 YOLOv5）
    for (const int class_id : class_set)
    {
        nms(valid_count, boxes, classes, order, class_id, nms_threshold);
    }

    // === 构建最终检测结果 ===
    for (int i = 0; i < valid_count && group->count < OBJ_NUMB_MAX_SIZE; ++i)
    {
        if (order[i] < 0) continue;  // 跳过被 NMS 抑制的框

        const int index = order[i];
        auto& result = group->results[group->count++];  // 获取下一个结果槽位并递增计数

        // 提取边界框坐标（模型输入空间）
        const float x1 = boxes[index * 4];
        const float y1 = boxes[index * 4 + 1];
        const float x2 = x1 + boxes[index * 4 + 2];
        const float y2 = y1 + boxes[index * 4 + 3];

        // === Letterbox 逆向映射 ===
        // letterbox 填充时在图像周围添加灰色边框以保持宽高比，
        // pad_x 为左侧（和右侧）的填充宽度，pad_y 为顶部（和底部）的填充高度。
        // 此步骤移除填充区域，将坐标还原到原始内容区域。
        const int content_w = std::max(1, static_cast<int>(model_in_w - 2.0F * pad_x));  // 内容实际宽度
        const int content_h = std::max(1, static_cast<int>(model_in_h - 2.0F * pad_y));  // 内容实际高度

        // 三步映射：
        // 1. 减去左侧/顶部填充偏移（移除 letterbox 的灰色边框区域）
        // 2. 裁剪到内容区域范围（防止浮点精度导致越界）
        // 3. 除以缩放因子映射回原始图像分辨率
        result.box.left   = static_cast<int>(clamp(x1 - pad_x, 0, content_w) / scale_w);
        result.box.top    = static_cast<int>(clamp(y1 - pad_y, 0, content_h) / scale_h);
        result.box.right  = static_cast<int>(clamp(x2 - pad_x, 0, content_w) / scale_w);
        result.box.bottom = static_cast<int>(clamp(y2 - pad_y, 0, content_h) / scale_h);

        result.class_id = classes[index];
        result.prop = probabilities[i];

        // 设置类别名称（与 post_process 相同的逻辑）
        const int id = classes[index];
        const std::string fallback_name = "class_" + std::to_string(id);
        const std::string& result_name =
            (id >= 0 && static_cast<size_t>(id) < labels.size()) ? labels[id] : fallback_name;
        strncpy(result.name, result_name.c_str(), OBJ_NAME_MAX_SIZE - 1);
        result.name[OBJ_NAME_MAX_SIZE - 1] = '\0';  // 强制终止符
    }
    return 0;
}
