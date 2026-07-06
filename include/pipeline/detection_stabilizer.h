#pragma once  // 头文件保护宏，防止重复包含

// 标准库头文件
#include <algorithm>  // std::max / std::min / std::clamp (C++17) 算法函数
#include <cmath>      // std::lround 四舍五入取整

// 项目内部头文件
#include "data_processing/postprocess.h"  // detect_result_group_t / BOX_RECT 检测结果结构

namespace pipeline {  // 管线命名空间

/**
 * @brief 计算两个检测框的 IoU（交并比）
 * @param a 第一个检测框
 * @param b 第二个检测框
 * @return IoU 值，范围 [0.0, 1.0]，值越大表示重叠越多
 *
 * IoU = 交集面积 / 并集面积。
 * 用于判断两个检测框是否检测到同一目标。
 */
inline float detectionIou(const BOX_RECT& a, const BOX_RECT& b) {
    const int left = std::max(a.left, b.left);        // 交集左边界 = 两框左边界中较大者
    const int top = std::max(a.top, b.top);            // 交集上边界 = 两框上边界中较大者
    const int right = std::min(a.right, b.right);      // 交集右边界 = 两框右边界中较小者
    const int bottom = std::min(a.bottom, b.bottom);   // 交集下边界 = 两框下边界中较小者
    const float intersection = static_cast<float>(
        std::max(0, right - left) * std::max(0, bottom - top));  // 交集面积（宽×高，clamp 到 0 防止负值）
    const float area_a = static_cast<float>(
        std::max(0, a.right - a.left) * std::max(0, a.bottom - a.top));  // 框 A 的面积
    const float area_b = static_cast<float>(
        std::max(0, b.right - b.left) * std::max(0, b.bottom - b.top));  // 框 B 的面积
    const float total = area_a + area_b - intersection;  // 并集面积 = A面积 + B面积 - 交集面积
    return total > 0.0F ? intersection / total : 0.0F;   // 避免除以零，返回 IoU
}

/**
 * @brief 平滑稳定检测结果（时域滤波）
 * @param previous   上一帧的检测结果（作为稳定参考）
 * @param current    当前帧的检测结果（将被修改为稳定后的值）
 * @param new_weight 当前帧权重，范围 [0.0, 1.0]，值越大当前帧影响越大（默认 0.45）
 *
 * 算法原理：
 *   1. 对 current 中的每个检测框，在 previous 中寻找 IoU > 0.25 的同类匹配
 *   2. 对匹配到的检测框，使用指数加权移动平均（EWMA）平滑其坐标
 *   3. 框坐标：稳定值 = previous * (1 - new_weight) + current * new_weight
 *   4. 置信度：取 max(current.prop, previous.prop * 0.92)，避免短暂遮挡导致置信度剧降
 *
 * 这能有效减少因视频噪声或轻微遮挡导致的检测框抖动。
 */
inline void stabilizeDetections(const detect_result_group_t& previous,
                                detect_result_group_t& current,
                                float new_weight = 0.45F) {
    new_weight = std::clamp(new_weight, 0.0F, 1.0F);  // 将权重限制在 [0, 1] 内，防止异常值
    for (int i = 0; i < current.count; ++i) {           // 遍历当前帧的每个检测结果
        auto& item = current.results[i];                // 当前检测项引用（将被修改）
        int best = -1;                                   // 最佳匹配的 previous 索引（-1 表示无匹配）
        float best_iou = 0.25F;                         // IoU 匹配阈值，需大于 0.25 才认为匹配
        for (int j = 0; j < previous.count; ++j) {      // 遍历上一帧的每个检测结果
            const auto& old = previous.results[j];       // 上一帧检测项
            if (old.class_id != item.class_id) continue; // 类别不同则跳过（必须同类才匹配）
            const float overlap = detectionIou(old.box, item.box); // 计算 IoU
            if (overlap > best_iou) {                    // 找到更高 IoU 的匹配
                best_iou = overlap;                      // 更新最佳 IoU
                best = j;                                // 记录匹配的索引
            }
        }
        if (best < 0) continue;                          // 无符合条件的匹配，跳过该检测框不做平滑

        const auto& old = previous.results[best];        // 最佳匹配的上一帧检测项
        // lambda：指数加权移动平均（EWMA），prior=旧值权重, next=新值权重
        const auto blend = [new_weight](int prior, int next) {
            return static_cast<int>(std::lround(
                prior * (1.0F - new_weight) + next * new_weight)); // EWMA 公式，四舍五入
        };
        item.box.left = blend(old.box.left, item.box.left);       // 平滑左边坐标
        item.box.top = blend(old.box.top, item.box.top);           // 平滑上边坐标
        item.box.right = blend(old.box.right, item.box.right);    // 平滑右边坐标
        item.box.bottom = blend(old.box.bottom, item.box.bottom); // 平滑下边坐标
        item.prop = std::max(item.prop, old.prop * 0.92F);        // 置信度衰减保护（不低于上一帧的 92%）
    }
}

}  // namespace pipeline
