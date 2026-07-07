#include "pipeline/mosaic_stream_pipeline.h"     // 马赛克流管线头文件
#include "pipeline/rga_preprocessor.h"          // RGA硬件预处理模块

#include <algorithm>     // std::min, std::max, std::clamp, std::sort
#include <array>         // std::array 固定大小数组
#include <cctype>        // std::isdigit, std::tolower 字符分类/转换
#include <chrono>        // 高精度时间测量
#include <cstring>       // std::strncmp 字符串比较
#include <cmath>         // std::lround 四舍五入取整
#include <cstdio>        // std::snprintf 格式化输出
#include <fstream>       // 文件读写（读取 /proc/stat、/sys 等）
#include <sstream>       // 字符串流（从文件读百分比值）
#include <iostream>      // 控制台输出
#include <tuple>         // std::tuple 多值打包
#include <vector>        // std::vector 动态数组

#include "rga.h"         // RGA 硬件加速 API（Rockchip 平台）
#include "im2d.h"        // RGA 2D 图像处理 API
#include "RgaUtils.h"    // RGA 工具函数

#include <opencv2/imgproc.hpp>   // OpenCV 图像处理（resize, putText 等）

#include "utils/thread_runtime.h"                // 线程运行时设置（CPU 亲缘性/优先级）
#include "pipeline/detection_stabilizer.h"       // 检测结果稳定器（去抖/滤波）

namespace pipeline {
namespace {

// ============================================================================
// TileGeometry 结构体：描述单个子画面在马赛克画布中的几何映射关系
// ============================================================================

struct TileGeometry {
    cv::Rect source;       // 源帧中的裁剪区域（用于裁切或缩放前的ROI选择）
    cv::Rect destination;  // 目标画布中的绘制区域（tile 在马赛克画布中的位置和尺寸）
};

// ============================================================================
// tileGeometry: 计算子画面在马赛克 tile 内的精确布局
// @param frame          输入的源帧
// @param tile           目标 tile 在马赛克画布中的位置和尺寸
// @param preserve_aspect 是否保持宽高比，true 则按比例缩放，false 则拉伸填满
// @param mode           缩放模式："cover"（裁切填满）或 "fit"（等比缩放留黑边）
// @return               TileGeometry 结构体，包含源 ROI 和目标 ROI
// ============================================================================
TileGeometry tileGeometry(int frame_width, int frame_height, const cv::Rect& tile,
                          bool preserve_aspect, const std::string& mode) {
    // 默认行为：源帧整体 → 目标 tile 拉伸填满（不保持宽高比）
    TileGeometry out{cv::Rect(0, 0, frame_width, frame_height), tile};
    if (!preserve_aspect || frame_width <= 0 || frame_height <= 0) return out;

    // ---- "cover" 模式：裁切源帧以填满目标 tile ----
    if (mode == "cover") {
        const double source_aspect = static_cast<double>(frame_width) / frame_height;
        const double target_aspect = static_cast<double>(tile.width) / tile.height; // 目标宽高比
        if (source_aspect > target_aspect) {
            // 源帧比目标更宽，需要裁切左右两侧
            // 计算需要保留的宽度 = 高度 × 目标宽高比，并对齐到偶数像素（RGA 要求）
            int crop_w = std::max(2, static_cast<int>(std::lround(frame_height * target_aspect)) & ~1);
            // 居中裁切
            out.source = cv::Rect(((frame_width - crop_w) / 2) & ~1, 0, crop_w, frame_height);
        } else {
            // 源帧比目标更高，需要裁切上下两侧
            int crop_h = std::max(2, static_cast<int>(std::lround(frame_width / target_aspect)) & ~1);
            out.source = cv::Rect(0, ((frame_height - crop_h) / 2) & ~1, frame_width, crop_h);
        }
        return out;
    }

    // ---- "fit" 模式：等比缩放，留黑边 ----
    // 计算缩放因子（取较小的方向，确保整帧可见）
    const double scale = std::min(
        static_cast<double>(tile.width) / frame_width,
        static_cast<double>(tile.height) / frame_height);
    // 计算缩放后的实际尺寸（对齐偶数像素）
    const int output_w = std::max(2, static_cast<int>(std::lround(frame_width * scale)) & ~1);
    const int output_h = std::max(2, static_cast<int>(std::lround(frame_height * scale)) & ~1);
    // 目标区域在 tile 中居中放置（左右/上下留黑边）
    out.destination = cv::Rect(tile.x + (tile.width - output_w) / 2,
                               tile.y + (tile.height - output_h) / 2,
                               output_w, output_h);
    return out;
}

TileGeometry tileGeometry(const cv::Mat& frame, const cv::Rect& tile,
                          bool preserve_aspect, const std::string& mode) {
    return tileGeometry(frame.cols, frame.rows, tile, preserve_aspect, mode);
}

// 断流后 FrameHub 会保留最后一帧以等待快速重连。合流端只允许短暂复用，
// 超时后把该源视为离线，避免冻结画面持续拉高端到端延迟指标。
void invalidateStaleSnapshots(std::vector<FrameHub::FrameSnapshot>& snapshots,
                              int stale_ms) {
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    const int64_t stale_ns = static_cast<int64_t>(std::max(1, stale_ms)) * 1000000LL;
    for (auto& item : snapshots) {
        if (item.capture_mono_ns <= 0 || now_ns - item.capture_mono_ns <= stale_ns) continue;
        item.frame.reset();
        item.dma = {};
        item.timestamp = {};
        item.capture_mono_ns = 0;
    }
}

cv::Rect logicalToRawRect(const cv::Rect& logical, int raw_width, int raw_height,
                          int rotation) {
    if (rotation == 90) {
        return {logical.y, raw_height - logical.x - logical.width,
                logical.height, logical.width};
    }
    if (rotation == 180) {
        return {raw_width - logical.x - logical.width,
                raw_height - logical.y - logical.height,
                logical.width, logical.height};
    }
    if (rotation == 270) {
        return {raw_width - logical.y - logical.height, logical.x,
                logical.height, logical.width};
    }
    return logical;
}

// ============================================================================
// readFirstPercent: 从文件中读取第一个数值（0.0-100.0）
// 用于读取 /sys/kernel/debug/rknpu/load、/sys/class/devfreq/dmc/load 等
// @param path 文件路径
// @return     解析到的百分比值，失败返回 0.0
// ============================================================================
double readFirstPercent(const char* path) {
    std::ifstream input(path);      // 打开文件
    if (!input) return 0.0;         // 文件不存在或无法打开
    // 将整个文件内容读入字符串
    std::string content((std::istreambuf_iterator<char>(input)),
                        std::istreambuf_iterator<char>());
    // 从字符串中查找第一个数字
    for (size_t i = 0; i < content.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(content[i]))) continue; // 跳过非数字字符
        size_t used = 0;
        try {
            // 尝试从当前位置解析浮点数
            const double value = std::stod(content.substr(i), &used);
            return std::clamp(value, 0.0, 100.0);   // 限制在 0-100 范围内
        } catch (...) {
            return 0.0; // 解析失败
        }
    }
    return 0.0; // 文件中没有找到数字
}

// ============================================================================
// readPercentAfter: 在文件中查找指定标记后的数值（0.0-100.0）
// 用于读取 /sys/kernel/debug/rkrga/load（格式为 "load = xx%"）
// @param path   文件路径
// @param marker 要查找的标记字符串（如 "load ="）
// @return       标记后的百分比值，失败返回 0.0
// ============================================================================
double readPercentAfter(const char* path, const std::string& marker) {
    std::ifstream input(path);      // 打开文件
    if (!input) return 0.0;         // 文件不存在
    std::string line;
    // 逐行读取
    while (std::getline(input, line)) {
        const size_t marker_pos = line.find(marker);  // 查找标记位置
        if (marker_pos == std::string::npos) continue; // 未找到标记，继续下一行
        // 从标记后开始扫描数字
        for (size_t i = marker_pos + marker.size(); i < line.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(line[i]))) continue; // 跳过非数字
            try { return std::clamp(std::stod(line.substr(i)), 0.0, 100.0); } // 解析并限制范围
            catch (...) { return 0.0; }
        }
    }
    return 0.0; // 未找到任何数值
}

// ============================================================================
// drawTileOverlay: 在画布上绘制单个 tile 的检测框叠加层
// 包括检测框、角标装饰、类别标签等可视化元素
// @param canvas     目标画布（直接绘入）
// @param overlay    帧叠加数据（包含检测结果）
// @param content    tile 内容在马赛克画布中的实际区域
// @param tile       原始 tile 边界（用于标签越界判断）
// @param source_roi 源帧中使用的 ROI（用于坐标映射）
// ============================================================================
void drawTileOverlay(cv::Mat& canvas, const FrameHub::FrameOverlay& overlay,
                     const cv::Rect& content, const cv::Rect& tile,
                     const cv::Rect& source_roi) {
    if (content.width <= 0 || content.height <= 0) return; // 无效区域，跳过

    // 计算源 ROI → 内容区域的坐标缩放比例（用于将检测框从源坐标系映射到画布坐标系）
    const double sx = static_cast<double>(content.width) /
        std::max(1, source_roi.width);   // X 方向缩放
    const double sy = static_cast<double>(content.height) /
        std::max(1, source_roi.height);  // Y 方向缩放

    // 各模型的检测框颜色（BGR 格式），最多支持 kMaxInferenceModels 个模型
    const std::array<cv::Scalar, kMaxInferenceModels> colors{{
        {180, 190, 55},   // 模型 0: 青绿色
        {70, 155, 235},   // 模型 1: 橙色
        {205, 105, 160},  // 模型 2: 粉紫色
        {180, 190, 55}}}; // 模型 3: 青绿色（同 0）

    // 根据画布大小自适应调整绘制参数
    const int box_thickness = std::clamp(canvas.cols / 900, 2, 3);       // 框线粗细
    const int text_thickness = 1;                                         // 文字粗细
    const double font_scale = std::clamp(canvas.cols / 2200.0, 0.45, 0.8); // 字体大小

    // 仅在检测结果有效时绘制框
    if (overlay.detections_valid) {
        // ---- 收集所有检测候选 ----
        // 遍历所有模型组的检测结果，构建候选列表
        struct Candidate {
            const detect_result_t* item = nullptr; // 检测项指针
            size_t group_index = 0;                 // 所属模型组索引
        };
        std::vector<Candidate> candidates;
        for (size_t group_index = 0; group_index < overlay.detections.size(); ++group_index) {
            const auto& group = overlay.detections[group_index];   // 获取该模型组的检测结果
            for (int i = 0; i < group.count; ++i) {
                candidates.push_back(Candidate{&group.results[i], group_index}); // 收集每个检测框
            }
        }

        // 按置信度从高到低排序（用于后续去重：高置信度保留）
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.item->prop > b.item->prop;
        });

        // ---- 跨模型去重（NMS 变体） ----
        // 各模型轮转推理后会同时缓存结果。这里做一次跨模型去重：同名目标重叠
        // 55% 即保留置信度最高者；异名框只有几乎完全重合时才压制，避免误删
        // person/helmet 等合理的嵌套目标。
        std::vector<Candidate> kept; // 保留的候选
        for (const Candidate& candidate : candidates) {
            bool duplicate = false;
            for (const Candidate& accepted : kept) {
                // 计算两个框的 IoU（交并比）
                const float overlap = detectionIou(candidate.item->box, accepted.item->box);
                // 判断是否为同类名目标
                const bool same_name = std::strncmp(candidate.item->name, accepted.item->name,
                                                    OBJ_NAME_MAX_SIZE) == 0;
                // 同名目标重叠 ≥55% 或异名目标重叠 ≥90% 视为重复
                if ((same_name && overlap >= 0.55F) || (!same_name && overlap >= 0.90F)) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) kept.push_back(candidate); // 非重复，保留
        }

        // ---- 绘制每个保留的检测框 ----
        for (const Candidate& candidate : kept) {
                const auto& item = *candidate.item;

                // 将检测框坐标从源坐标系映射到画布坐标系
                int left = content.x + static_cast<int>(
                    std::lround((item.box.left - source_roi.x) * sx));     // 左边界
                int top = content.y + static_cast<int>(
                    std::lround((item.box.top - source_roi.y) * sy));      // 上边界
                int right = content.x + static_cast<int>(
                    std::lround((item.box.right - source_roi.x) * sx));    // 右边界
                int bottom = content.y + static_cast<int>(
                    std::lround((item.box.bottom - source_roi.y) * sy));   // 下边界

                // 将坐标裁剪到内容区域范围内，防止越界
                left = std::clamp(left, content.x, content.x + content.width - 1);
                right = std::clamp(right, content.x, content.x + content.width - 1);
                top = std::clamp(top, content.y, content.y + content.height - 1);
                bottom = std::clamp(bottom, content.y, content.y + content.height - 1);

                if (right <= left || bottom <= top) continue; // 无效框，跳过

                const cv::Scalar color = colors[candidate.group_index]; // 使用对应模型的颜色

                // 1）绘制主检测框（矩形）
                cv::rectangle(canvas, {left, top}, {right, bottom}, color,
                              box_thickness, cv::LINE_AA);

                // 2）绘制四个角的装饰线（L 形角标，增强视觉辨识度）
                const int corner = std::clamp(std::min(right - left, bottom - top) / 7, 10, 24); // 角线长度
                const int accent = std::min(4, box_thickness + 1); // 角线粗细（比主框略粗）

                // 左上角
                cv::line(canvas, {left, top}, {left + corner, top}, color, accent, cv::LINE_AA);
                cv::line(canvas, {left, top}, {left, top + corner}, color, accent, cv::LINE_AA);
                // 右上角
                cv::line(canvas, {right, top}, {right - corner, top}, color, accent, cv::LINE_AA);
                cv::line(canvas, {right, top}, {right, top + corner}, color, accent, cv::LINE_AA);
                // 左下角
                cv::line(canvas, {left, bottom}, {left + corner, bottom}, color, accent, cv::LINE_AA);
                cv::line(canvas, {left, bottom}, {left, bottom - corner}, color, accent, cv::LINE_AA);
                // 右下角
                cv::line(canvas, {right, bottom}, {right - corner, bottom}, color, accent, cv::LINE_AA);
                cv::line(canvas, {right, bottom}, {right, bottom - corner}, color, accent, cv::LINE_AA);

                // 3）绘制标签（类别名 + 置信度百分比）
                char label[64];
                std::snprintf(label, sizeof(label), "%s %.0f%%", item.name, item.prop * 100.0f); // 构造标签
                int baseline = 0;
                const cv::Size text = cv::getTextSize( // 计算文字渲染尺寸
                    label, cv::FONT_HERSHEY_SIMPLEX, font_scale, text_thickness, &baseline);

                // 标签水平位置：框的左上角，但不超出 tile 右边界
                const int label_x = std::clamp(left, tile.x, tile.x + tile.width - text.width - 8);
                // 标签垂直位置：默认在框上方，如果上方空间不足则放在框下方
                int label_top = top - text.height - baseline - 8;
                if (label_top < tile.y + 40) label_top = std::min(bottom + 2,
                    tile.y + tile.height - text.height - baseline - 8);

                // 标签背景区域
                const cv::Rect background(label_x, label_top,
                    std::min(text.width + 8, tile.x + tile.width - label_x),
                    text.height + baseline + 8);

                // 仅在标签完全在 tile 内时绘制（避免文字被截断）
                if (background.width > 0 && background.height > 0 &&
                    background.y >= tile.y && background.y + background.height <= tile.y + tile.height) {
                    // 背景底色：深色半透明
                    cv::rectangle(canvas, background, {22, 25, 29}, cv::FILLED, cv::LINE_AA);
                    // 左侧色条：模型对应颜色（增强视觉关联）
                    cv::rectangle(canvas, {background.x, background.y, 3, background.height},
                                  color, cv::FILLED, cv::LINE_AA);
                    // 标签文字：白色
                    cv::putText(canvas, label,
                        {label_x + 4, label_top + text.height + 2},
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, {255, 255, 255},
                        text_thickness, cv::LINE_AA);
                }
        }
    }

}

}  // namespace

/**
 * @file mosaic_stream_pipeline.cpp
 * @brief 马赛克流处理管道实现
 *
 * 该文件实现了 MosaicStreamPipeline 类，负责将多路视频流合成为一个马赛克画面，
 * 并将合成后的视频流通过 RTSP 推送出去。核心流程包括：
 *
 * 1. 从 FrameHub 获取多路源帧（ExternalRtsp + Imx415）
 * 2. 通过 RGA 硬件加速进行 resize + blit 操作，将各路帧拼入画布
 * 3. 绘制检测框叠加层（包含跨模型去重、角标装饰、标签）
 * 4. 绘制全局遥测信息叠加层（FPS、延迟、CPU/NPU/RGA/DDR 使用率）
 * 5. 通过 EncodedMediaService 进行 H264 编码和 RTSP 推流
 * 6. 支持录像触发和归档
 */

/**
 * @brief 构造函数，初始化马赛克流处理管道
 * @param config 马赛克流配置参数（分辨率、帧率、码率、RTSP地址等）
 * @param hub    帧中心共享指针，用于获取各路视频流的最新帧
 *
 * 初始化马赛克流处理管道，保存配置参数和帧中心引用。
 * 注意：构造函数不启动管线，需调用 start() 方法。
 */
MosaicStreamPipeline::MosaicStreamPipeline(MosaicStreamConfig config, std::shared_ptr<FrameHub> hub)
    : config_(std::move(config)), hub_(std::move(hub)) {}

/**
 * @brief 析构函数，安全停止马赛克流处理管道
 *
 * 调用 stop() 方法停止工作线程、释放 RGA 资源、关闭媒体服务。
 * 确保所有资源在对象销毁前被正确清理。
 */
MosaicStreamPipeline::~MosaicStreamPipeline() {
    stop(); // 先停止管线再析构
}

/**
 * @brief 从 INI 配置文件加载马赛克流配置
 * @param path 配置文件路径
 * @param out  输出的配置结构体（MosaicStreamConfig 引用）
 * @return     是否加载成功
 *
 * 从 INI 配置文件加载马赛克流配置，包括：
 * - [mosaic_stream] 节：分辨率、帧率、码率、输入模式等
 * - [stream_output] 节：RTSP 推送地址
 * - [record] 节：录像参数（预录/后录时长、缓存大小等）
 * - [sync] 节：多源帧同步参数
 */
bool MosaicStreamPipeline::loadFromIni(const std::string& path, MosaicStreamConfig& out) {
    IniConfig cfg;
    if (!cfg.load(path)) {
        return false; // 配置文件加载失败
    }

    // ---- [mosaic_stream] 节：马赛克流基本参数 ----
    out.enable = cfg.getBool("mosaic_stream", "enable", false);              // 是否启用马赛克流
    out.width = cfg.getInt("mosaic_stream", "width", 1280);                  // 输出画面宽度
    out.height = cfg.getInt("mosaic_stream", "height", 720);                 // 输出画面高度
    out.fps = cfg.getInt("mosaic_stream", "fps", 15);                        // 输出帧率
    out.bitrate = cfg.getInt("mosaic_stream", "bitrate", 2000000);           // 编码码率（bps）
    out.enable_rtsp = cfg.getBool("mosaic_stream", "enable_rtsp", true);     // 是否启用 RTSP 推流
    // RTSP 推送地址，从 [stream_output] 节读取
    out.rtsp_url = cfg.getString("stream_output", "url", "rtsp://192.168.137.1:8554/result");
    out.input_mode = cfg.getString("mosaic_stream", "input_mode", "side_by_side"); // 输入模式：side_by_side 或 vertical
    out.preserve_aspect_ratio = cfg.getBool(                                 // 是否保持宽高比
        "mosaic_stream", "preserve_aspect_ratio", out.preserve_aspect_ratio);
    out.aspect_mode = cfg.getString("mosaic_stream", "aspect_mode", out.aspect_mode); // 宽高比处理模式：cover 或 fit
    out.draw_overlay = cfg.getBool("mosaic_stream", "draw_overlay", out.draw_overlay); // 是否绘制检测框叠加层
    out.stream_queue_size = cfg.getInt("mosaic_stream", "stream_queue_size", out.stream_queue_size); // 流队列大小
    out.rtsp_packet_queue_size = cfg.getInt(                                 // RTSP 包队列大小
        "mosaic_stream", "rtsp_packet_queue_size", out.rtsp_packet_queue_size);
    out.stream_drop_oldest_when_full = cfg.getBool(                           // 队列满时是否丢弃最旧帧
        "mosaic_stream", "stream_drop_oldest_when_full", out.stream_drop_oldest_when_full);
    out.stream_consume_latest_only = cfg.getBool(                             // 是否仅消费最新帧
        "mosaic_stream", "stream_consume_latest_only", out.stream_consume_latest_only);
    out.stream_resize_on_mismatch = cfg.getBool(                              // 尺寸不匹配时是否自动缩放
        "mosaic_stream", "stream_resize_on_mismatch", out.stream_resize_on_mismatch);

    // ---- [record] 节：录像参数 ----
    out.recorder.enabled = cfg.getBool("record", "enabled", false);          // 是否启用录像
    out.recorder.pre_seconds = cfg.getInt("record", "pre_seconds", out.recorder.pre_seconds);   // 预录秒数
    out.recorder.post_seconds = cfg.getInt("record", "post_seconds", out.recorder.post_seconds); // 后录秒数
    out.recorder.cache_seconds = cfg.getInt("record", "cache_seconds", out.recorder.cache_seconds); // 缓存秒数
    out.recorder.packet_queue_size = cfg.getInt("record", "packet_queue_size", out.recorder.packet_queue_size); // 包队列大小
    out.recorder.confirm_window = cfg.getInt("record", "confirm_window", out.recorder.confirm_window); // 确认窗口大小
    out.recorder.confirm_min_positive = cfg.getInt(                           // 确认最少阳性数
        "record", "confirm_min_positive", out.recorder.confirm_min_positive);
    out.recorder.output_dir = cfg.getString("record", "output_dir", out.recorder.output_dir); // 录像输出目录
    out.recorder.max_files = cfg.getInt("record", "max_files", out.recorder.max_files);       // 最大保留文件数
    out.recorder.max_storage_mb = cfg.getInt("record", "max_storage_mb", out.recorder.max_storage_mb); // 最大存储空间（MB）
    out.recorder.min_free_space_mb = cfg.getInt(                              // 最小保留空闲空间（MB）
        "record", "min_free_space_mb", out.recorder.min_free_space_mb);

    // ---- [sync] 节：多源帧同步 ----
    out.sync_enabled = cfg.getBool("sync", "enabled", out.sync_enabled);      // 是否启用帧同步
    out.sync_threshold_ms = cfg.getInt("sync", "threshold_ms", out.sync_threshold_ms); // 同步阈值（毫秒）
    out.sync_queue_depth = cfg.getInt("sync", "queue_depth", out.sync_queue_depth); // 同步队列深度
    out.source_stale_ms = cfg.getInt("sync", "source_stale_ms", out.source_stale_ms); // 断流旧帧失效阈值

    // 将 input_mode 转为小写，统一格式
    std::transform(out.input_mode.begin(), out.input_mode.end(), out.input_mode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    // 仅支持 side_by_side（左右拼接）和 vertical（上下拼接）两种模式
    if (out.input_mode != "side_by_side" && out.input_mode != "vertical") {
        out.input_mode = "side_by_side"; // 非法模式回退到默认
    }

    return true;
}

/**
 * @brief 启动马赛克流处理管道
 * @return 是否启动成功
 *
 * 按以下顺序启动：
 * 1. 幂等检查：已在运行则直接返回
 * 2. FrameHub 验证：确保帧中心已正确装配
 * 3. 配置验证：RTSP 地址等
 * 4. 创建并启动 EncodedMediaService（H264 编码 + RTSP 推流）
 * 5. 启动合成工作线程 worker_
 */
bool MosaicStreamPipeline::start() {
    // 步骤 1: 幂等性检查
    if (running_) {
        return true; // 已在运行，直接返回成功
    }

    // 步骤 2: FrameHub 可用性验证
    // Mosaic 只负责汇总 external_rtsp 与 imx415，并统一外发；
    // 所以这里第一件事就是检查 FrameHub 是否已由 AppInitializer 正确装配。
    if (!hub_) {
        std::cerr << "MosaicStreamPipeline: frame hub not set" << std::endl;
        return false;
    }

    // 步骤 3: 设置同步队列深度
    hub_->setSyncQueueDepth(static_cast<size_t>(std::max(2, config_.sync_queue_depth)));

    // 步骤 4: RTSP 配置验证
    if (config_.enable_rtsp && config_.rtsp_url.empty()) {
        std::cerr << "MosaicStreamPipeline: RTSP enabled but rtsp_url is empty" << std::endl;
        config_.enable_rtsp = false; // 地址为空，禁用 RTSP
    }

    // 步骤 5: 构建媒体服务配置
    EncodedMediaConfig media_config;
    media_config.width = config_.width;                   // 编码宽度
    media_config.height = config_.height;                 // 编码高度
    media_config.fps = config_.fps;                       // 编码帧率
    media_config.bitrate = config_.bitrate;               // 编码码率
    media_config.enable_rtsp = config_.enable_rtsp;       // 是否推流
    media_config.rtsp_url = config_.rtsp_url;             // RTSP 地址
    media_config.frame_queue_size = config_.stream_queue_size; // 帧队列大小
    media_config.rtsp_packet_queue_size = config_.rtsp_packet_queue_size; // RTSP 包队列
    media_config.recorder = config_.recorder;             // 录像配置

    // 步骤 6: 创建 EncodedMediaService 实例
    media_service_ = std::make_unique<EncodedMediaService>(std::move(media_config));

    // 步骤 7: 设置录像完成回调（如果有）
    if (recording_completion_callback_) {
        media_service_->setRecordingCompletionCallback(recording_completion_callback_);
    }

    // 步骤 8: 启动媒体服务（内部启动编码线程和 RTSP 推流线程）
    if (!media_service_->start()) {
        std::cerr << "MosaicStreamPipeline: media service init failed" << std::endl;
        media_service_.reset(); // 启动失败，释放资源
        return false;
    }

    // 步骤 9: 设置运行标志并启动合成线程
    running_ = true;
    worker_ = std::thread([this]() { loop(); }); // 启动合成主循环线程

    return true;
}

/**
 * @brief 停止马赛克流处理管道
 *
 * 按顺序执行：设置停止标志 → 等待工作线程退出 → 停止媒体服务 → 释放资源
 * 线程安全的停止流程，可以安全地重复调用。
 */
void MosaicStreamPipeline::stop() {
    if (!running_) {
        return; // 已在停止状态
    }

    running_ = false; // 第一步：设置停止标志，通知工作线程退出

    // 第二步：等待工作线程完成最后一帧后退出
    if (worker_.joinable()) {
        worker_.join();
    }

    // 第三步：停止媒体服务（编码 + RTSP 推流）
    if (media_service_) media_service_->stop();
    media_service_.reset(); // 释放 EncodedMediaService 对象
}

// setStreamingEnabled: 运行时切换 RTSP 推流开关
// 注意：此设置在下一次重启时才会生效，不会立即影响当前推流状态
void MosaicStreamPipeline::setStreamingEnabled(bool enable) {
    config_.enable_rtsp = enable;
    if (running_.load()) {
        std::cerr << "MosaicStreamPipeline: RTSP mode change takes effect after restart" << std::endl;
    }
}

// triggerRecording: 触发一次录像事件
// @param event 录像事件（包含事件类型、时间戳等）
void MosaicStreamPipeline::triggerRecording(const recording::Event& event) {
    if (media_service_) media_service_->trigger(event);
}

// updateDetection: 更新检测状态（用于录像触发判断）
// @param type      事件类型
// @param detected  是否检测到目标
// @param mono_ns   单调时钟纳秒时间戳
// @param real_ms   真实时间毫秒
// @param camera_id 摄像头 ID
void MosaicStreamPipeline::updateDetection(recording::EventType type, bool detected,
                                           int64_t mono_ns, int64_t real_ms, int camera_id) {
    if (media_service_) media_service_->updateDetection(type, detected, mono_ns, real_ms, camera_id);
}

// setRecordingCompletionCallback: 设置录像完成回调
void MosaicStreamPipeline::setRecordingCompletionCallback(
    recording::EventRecorder::CompletionCallback callback) {
    recording_completion_callback_ = std::move(callback);
    if (media_service_) media_service_->setRecordingCompletionCallback(recording_completion_callback_);
}

/**
 * @brief 马赛克流处理主循环（运行在 worker_ 线程中）
 *
 * 使用 FrameHub 的通知机制替代固定 sleep，当任何帧源有新数据时立即触发合成。
 * 主要流程：
 * 1. 按输出帧率限制合成频率（frame_period）
 * 2. 使用 waitForNew 等待任意帧源的新帧到达
 * 3. 调用 composeFrame() 执行实际拼图合成
 * 4. 将合成帧提交给 media_service_ 进行编码推流
 */
void MosaicStreamPipeline::loop() {
    // 设置线程运行时属性：CPU 亲缘性绑定到 "mosaic" 组、"mosaic-loop" 角色
    utils::applyThreadRuntime("mosaic", "mosaic-loop");

    // 计算帧间隔（纳秒）：1秒 / 帧率
    const auto frame_period = std::chrono::nanoseconds(
        1000000000LL / std::max(1, config_.fps));

    // 需要等待的帧源列表：外部 RTSP 流 + IMX415 摄像头
    const std::vector<FrameSource> wait_sources{
        FrameSource::ExternalRtsp,  // 外部 RTSP 流（可见光）
        FrameSource::Imx415,        // IMX415 摄像头（红外/热成像）
    };

    // 每个源最近处理的帧序号（用于判断是否有新帧）
    std::vector<uint64_t> last_seq(wait_sources.size(), 0);

    // 下次允许合成的时间点（帧率节流）
    auto next_emit = std::chrono::steady_clock::now();

    while (running_) {
        // ---- 多源等待策略 ----
        // 任何一路出现新帧，都可以触发一次拼图刷新；
        // 同时按源分别等待，避免单一路卡死把整个 mosaic 节奏拖住。
        bool has_new = false;

        // 先看序号再等待，避免依次等待两路给新鲜帧额外增加半帧以上延迟
        std::vector<FrameHub::FrameSnapshot> seq_snapshot;
        hub_->snapshot(wait_sources, seq_snapshot); // 获取各源当前快照序号
        for (size_t i = 0; i < wait_sources.size(); ++i) {
            if (seq_snapshot[i].seq > last_seq[i]) has_new = true; // 任意源序号增加即视为有新帧
        }

        // 如果尚未检测到新帧，逐个等待（每路最多等 2ms，确保低延迟响应）
        const auto per_source_wait = std::chrono::milliseconds(2);
        for (size_t i = 0; i < wait_sources.size() && !has_new; ++i) {
            // 只要有任一路有新帧，就继续检查其他路，最后一起触发拼图刷新
            has_new = hub_->waitForNew(wait_sources[i], last_seq[i], per_source_wait) || has_new;
        }

        // 刷新各源最新序号
        hub_->snapshot(wait_sources, seq_snapshot);
        for (size_t i = 0; i < wait_sources.size(); ++i) {
            last_seq[i] = seq_snapshot[i].seq;
        }

        // 如果仍然没有新帧，继续循环等待
        if (!has_new) {
            continue;
        }

        // ---- 帧率节流 ----
        // 多路源可能合计产生 60fps 以上的更新。严格按输出 fps 合成，防止
        // 重复拼接最终会被 latest-only 编码队列丢掉的中间帧。
        const auto now = std::chrono::steady_clock::now();
        if (now < next_emit) {
            std::this_thread::sleep_until(next_emit); // 还没到下一次合成时间，休眠等待
        } else if (now - next_emit > frame_period) {
            // 处理超时后直接追到当前周期，避免延迟按帧持续累积
            next_emit = now;
        }
        next_emit += frame_period; // 更新下次合成时间

        // ---- 合成马赛克帧 ----
        // 真正的拼图动作集中在 composeFrame()，这里保持主循环尽量薄。
        // 好处是后续如果要做"拼图算法替换 / 叠字 / 时间戳 / 版式切换"，
        // 只需聚焦 composeFrame() 一处。
        ComposedFrame composed;
        const bool direct = composeDirectFrame(composed);
        if (!direct) continue;

        // ---- 提交合成帧 ----
        if (media_service_ && media_service_->active()) {
            output_rate_.tick(); // 记录输出速率
            int64_t mono_ns = last_composed_mono_ns_; // 单调时钟纳秒时间戳
            int64_t real_ms = last_composed_real_ms_; // 真实时间毫秒时间戳
            if (mono_ns == 0) {
                // 降级：使用当前时间作为时间戳
                mono_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
            }
            // 提交帧到媒体服务（内部执行 H264 编码 → RTSP 推流 / 录像归档）
            composed.capture_mono_ns = mono_ns;
            composed.capture_real_ms = real_ms;
            composed.frame_id = ++composed_frame_id_;
            media_service_->submitFrame(std::move(composed));
        }
    }
}

bool MosaicStreamPipeline::composeDirectFrame(ComposedFrame& output) {
    if (!hub_) return false;
    const std::vector<FrameSource> sources{
        FrameSource::ExternalRtsp, FrameSource::Imx415};
    std::vector<FrameHub::FrameSnapshot> snapshot;

    if (config_.sync_enabled) {
        FrameHub::SyncResult pair;
        if (hub_->takeSynchronized(
                sources[0], sources[1],
                static_cast<int64_t>(std::max(1, config_.sync_threshold_ms)) * 1000000LL,
                pair)) {
            snapshot = {std::move(pair.first), std::move(pair.second)};
            ++sync_pairs_;
            sync_fail_streak_ = 0;
            sync_skip_cooldown_ = 0;
            sync_drop_first_ += pair.dropped_first;
            sync_drop_second_ += pair.dropped_second;
            const int64_t delta = std::llabs(pair.delta_ns);
            sync_delta_total_ns_ += delta;
            sync_delta_max_ns_ = std::max(sync_delta_max_ns_, delta);
            last_synced_snapshot_ = snapshot;
            if (sync_pairs_ % 100 == 0) {
                std::cout << "[FrameSync] pairs=" << sync_pairs_
                          << " avg_ms=" << (sync_delta_total_ns_ / sync_pairs_ / 1000000.0)
                          << " max_ms=" << (sync_delta_max_ns_ / 1000000.0)
                          << " drops=" << sync_drop_first_ << ',' << sync_drop_second_ << '\n';
            }
        } else {
            sync_drop_first_ += pair.dropped_first;
            sync_drop_second_ += pair.dropped_second;
            ++sync_fail_streak_;
            if (last_synced_snapshot_.size() == sources.size()) {
                const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                int64_t newest_ns = 0;
                for (const auto& item : last_synced_snapshot_) {
                    newest_ns = std::max(newest_ns, item.capture_mono_ns);
                }
                // 短暂无新配对时重复上一组配对帧，不能退回“两路各取最新”。
                // 超过500ms则认为源已停滞，回退当前快照以便离线状态及时显现。
                if (newest_ns > 0 && now_ns - newest_ns <= 500000000LL) {
                    snapshot = last_synced_snapshot_;
                } else {
                    last_synced_snapshot_.clear();
                }
            }
            if (snapshot.empty()) hub_->snapshot(sources, snapshot);
        }
    } else {
        hub_->snapshot(sources, snapshot);
    }
    if (snapshot.size() != sources.size()) return false;

    invalidateStaleSnapshots(snapshot, config_.source_stale_ms);

    updateTelemetry(snapshot);
    last_composed_mono_ns_ = 0;
    last_composed_real_ms_ = 0;
    for (const auto& item : snapshot) {
        if (item.capture_mono_ns > 0) {
            last_composed_mono_ns_ = last_composed_mono_ns_ == 0
                ? item.capture_mono_ns
                : std::min(last_composed_mono_ns_, item.capture_mono_ns);
        }
        last_composed_real_ms_ = std::max(last_composed_real_ms_,
            std::chrono::duration_cast<std::chrono::milliseconds>(
                item.timestamp.time_since_epoch()).count());
    }

    std::array<bool, 2> available{{snapshot[0].dma.valid(), snapshot[1].dma.valid()}};
    const int available_count = static_cast<int>(available[0]) + static_cast<int>(available[1]);
    if (available_count == 0) return false;

    struct OverlayItem {
        FrameHub::FrameOverlay overlay;
        cv::Rect content;
        cv::Rect tile;
        cv::Rect source;
    };
    std::vector<OverlayItem> overlays;
    overlays.reserve(2);
    std::vector<std::pair<cv::Rect, std::string>> offline_tiles;

    const int rows = config_.input_mode == "vertical" ? 2 : 1;
    const int cols = config_.input_mode == "vertical" ? 1 : 2;
    const int tile_w = config_.width / cols;
    const int tile_h = config_.height / rows;
    for (int i = 0; i < 2; ++i) {
        // 源离线时仍保留固定槽位。自动把单路竖画面扩到整张横屏，会产生巨大的
        // 左右黑边，也会让来源位置在重连瞬间跳变，运维端很难判断哪一路掉线。
        const cv::Rect tile{
            (i % cols) * tile_w, (i / cols) * tile_h,
            i % cols == cols - 1 ? config_.width - (i % cols) * tile_w : tile_w,
            i / cols == rows - 1 ? config_.height - (i / cols) * tile_h : tile_h};
        if (!available[i]) {
            offline_tiles.emplace_back(
                tile, i == 0 ? "DIRECT INPUT OFFLINE" : "IMX415 OFFLINE");
            continue;
        }
        const auto& dma = snapshot[i].dma;
        TileGeometry geometry = tileGeometry(
            dma.logicalWidth(), dma.logicalHeight(), tile,
            config_.preserve_aspect_ratio, config_.aspect_mode);
        // NV12目标ROI必须偶数对齐；居中偏移也向下对齐，最多改变1像素。
        geometry.destination.x &= ~1;
        geometry.destination.y &= ~1;
        geometry.destination.width &= ~1;
        geometry.destination.height &= ~1;
        cv::Rect raw_source = logicalToRawRect(
            geometry.source, dma.width, dma.height, dma.rotation);
        raw_source.x &= ~1;
        raw_source.y &= ~1;
        raw_source.width &= ~1;
        raw_source.height &= ~1;

        RgaDmaComposeTask task;
        task.src_fd = dma.fd;
        task.src_width = dma.width;
        task.src_height = dma.height;
        task.src_width_stride = dma.width_stride;
        task.src_height_stride = dma.height_stride;
        task.src_buffer_size = dma.buffer_size;
        task.src_format = dma.format == DmaPixelFormat::NV21
            ? RgaPixelFormat::NV21
            : dma.format == DmaPixelFormat::NV16
                ? RgaPixelFormat::NV16 : RgaPixelFormat::NV12;
        if (dma.color_space == DmaColorSpace::Bt601Full) {
            task.color_space = RgaColorSpace::Bt601Full;
        } else if (dma.color_space == DmaColorSpace::Bt601Limited) {
            task.color_space = RgaColorSpace::Bt601Limited;
        } else {
            task.color_space = RgaColorSpace::Bt709Limited;
        }
        task.rotation = dma.rotation;
        task.source_rect = raw_source;
        task.destination_rect = geometry.destination;
        task.lease = dma.lease;
        output.dma_layers.push_back(std::move(task));
        cv::Rect detection_source = geometry.source;
        const int detection_width = snapshot[i].overlay.source_width;
        const int detection_height = snapshot[i].overlay.source_height;
        if (detection_width > 0 && detection_height > 0 &&
            (detection_width != dma.logicalWidth() || detection_height != dma.logicalHeight())) {
            const double sx = static_cast<double>(detection_width) / dma.logicalWidth();
            const double sy = static_cast<double>(detection_height) / dma.logicalHeight();
            detection_source = {
                static_cast<int>(std::lround(geometry.source.x * sx)),
                static_cast<int>(std::lround(geometry.source.y * sy)),
                std::max(1, static_cast<int>(std::lround(geometry.source.width * sx))),
                std::max(1, static_cast<int>(std::lround(geometry.source.height * sy)))};
        }
        overlays.push_back(OverlayItem{
            snapshot[i].overlay, geometry.destination, tile, detection_source});
    }
    if (output.dma_layers.empty()) return false;

    if (config_.draw_overlay) {
        char status[256];
        const double encoded_fps = media_service_ ? media_service_->encodedFps() : 0.0;
        const double sent_fps = media_service_ ? media_service_->sentFps() : 0.0;
        const double out_fps = sent_fps > 0.0 ? sent_fps : encoded_fps;
        const int64_t latency = media_service_ ? media_service_->lastFrameAgeMs() : 0;
        std::snprintf(status, sizeof(status),
            "%.1f FPS  |  %lld ms  |  DROP %.2f%%  |  CPU %.0f%%  NPU %.0f%%  RGA %.0f%%  DDR %.0f%%",
            out_fps, static_cast<long long>(latency), telemetry_.drop_percent,
            telemetry_.cpu_percent, telemetry_.npu_percent,
            telemetry_.rga_percent, telemetry_.ddr_percent);
        const std::string status_text(status);
        const bool vertical_layout = config_.input_mode == "vertical";
        output.draw_luma_overlay = [items = std::move(overlays),
                                    offline = std::move(offline_tiles),
                                    status_text, vertical_layout](cv::Mat& luma) {
            // 一像素低对比分隔线让两路边界清楚，但不抢夺检测框视觉焦点。
            if (vertical_layout) {
                cv::line(luma, {0, luma.rows / 2}, {luma.cols, luma.rows / 2},
                         cv::Scalar(88), 1, cv::LINE_8);
            } else {
                cv::line(luma, {luma.cols / 2, 0}, {luma.cols / 2, luma.rows},
                         cv::Scalar(88), 1, cv::LINE_8);
            }
            for (const auto& item : items) {
                drawTileOverlay(luma, item.overlay, item.content, item.tile, item.source);
            }
            for (const auto& item : offline) {
                const cv::Rect& tile = item.first;
                const std::string& label = item.second;
                int baseline = 0;
                const double scale = std::clamp(tile.width / 1800.0, 0.45, 0.7);
                const cv::Size size = cv::getTextSize(
                    label, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
                cv::putText(luma, label,
                    {tile.x + std::max(8, (tile.width - size.width) / 2),
                     tile.y + tile.height / 2},
                    cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(150), 1, cv::LINE_AA);
            }
            const double font_scale = std::clamp(luma.cols / 3000.0, 0.46, 0.64);
            int baseline = 0;
            const cv::Size text_size = cv::getTextSize(
                status_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);
            const cv::Rect bar(0, 0, luma.cols, text_size.height + baseline + 14);
            cv::rectangle(luma, bar, cv::Scalar(16), cv::FILLED, cv::LINE_8);
            cv::line(luma, {0, bar.height - 1}, {luma.cols, bar.height - 1},
                     cv::Scalar(180), 1, cv::LINE_8);
            cv::putText(luma, status_text,
                {(luma.cols - text_size.width) / 2, text_size.height + 4},
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(235), 1, cv::LINE_AA);
        };
    }
    return true;
}

/**
 * @brief RGA 批量处理：resize + 直接拼接（一次性提交所有任务给 RGA 硬件）
 * @param tasks  任务列表，每个元素为 (源帧, 目标宽度, 目标高度, 目标 ROI)
 * @param canvas 目标画布（直接写入）
 * @return       true 表示所有 RGA 操作成功完成
 *
 * 优化方案 A + B：
 * - 方案 A：使用 imbeginJob/imendJob 批量提交所有 resize 任务，减少 IOCTL 调用次数
 * - 方案 B：使用 imtranslateTask 直接把 resize 后的 tile 写入画布 ROI，省去 CPU copyTo
 *
 * RGA 批量处理可以显著降低 CPU 开销：原本 N 个 tile 需要 N 次内核调用，
 * 现在合并为 1 次批量提交。
 */
bool MosaicStreamPipeline::rgaBatchCompose(
    const std::vector<std::tuple<cv::Mat, cv::Rect, cv::Rect>>& tasks,
    cv::Mat& canvas) {
    // 前置检查：RGA 可用性、运行时健康状态、任务非空
    if (!rga_available_ || !rgaRuntimeHealthy() || tasks.empty()) {
        return false;
    }
    RgaSubmissionGuard rga_guard; // RGA 提交守卫：串行化 RGA 硬件访问
    if (!rgaRuntimeHealthy()) return false; // 再次检查（守卫可能因故障禁用了 RGA）

    // 1. 创建 RGA 任务批处理句柄
    // imbeginJob 创建一个批处理会话，后续的所有 Task API 都会累积到此会话中
    im_job_handle_t job = imbeginJob(0);
    if (job == 0) {
        // 某些旧 BSP 不支持 task API；让调用方退回单任务 RGA，不影响全局 RGA。
        return false;
    }

    bool success = true;
    std::vector<cv::Mat> continuous_srcs; // 保持连续源帧的生命周期（防止悬空指针）
    continuous_srcs.reserve(tasks.size());

    // 2. 准备画布的 RGA 缓冲描述
    rga_buffer_t rga_canvas{};
    rga_canvas.vir_addr = canvas.data;       // 画布虚拟地址（CPU 可访问）
    rga_canvas.width = canvas.cols;          // 画布宽度（像素）
    rga_canvas.height = canvas.rows;         // 画布高度（像素）
    rga_canvas.wstride = static_cast<int>(canvas.step / canvas.elemSize());
    rga_canvas.hstride = canvas.rows;        // 列步长
    rga_canvas.format = RK_FORMAT_BGR_888;   // BGR 8-8-8 格式

    // 3. 每路用一个 process task 直接完成 crop + resize + 写入画布 ROI。
    // 不创建 tile 中间图，也不再追加 translate task。
    for (const auto& task : tasks) {
        const cv::Mat& src = std::get<0>(task);   // 源帧
        const cv::Rect& source_roi = std::get<1>(task);
        const cv::Rect& destination_roi = std::get<2>(task);

        if (src.empty() || src.type() != CV_8UC3 || source_roi.area() <= 0 ||
            destination_roi.area() <= 0 ||
            (source_roi & cv::Rect(0, 0, src.cols, src.rows)) != source_roi ||
            (destination_roi & cv::Rect(0, 0, canvas.cols, canvas.rows)) != destination_roi) {
            success = false;
            break;
        }

        // 确保源帧内存连续（RGA 需要连续内存布局）
        cv::Mat continuous_src;
        if (!src.isContinuous() || src.data != src.datastart) {
            continuous_src = src.clone(); // 非连续则克隆为连续
        } else {
            continuous_src = src;
        }
        continuous_srcs.push_back(continuous_src); // 保持引用，防止被释放

        // 配置源缓冲描述
        rga_buffer_t rga_src{};
        rga_src.vir_addr = const_cast<void*>(static_cast<const void*>(continuous_src.data));
        rga_src.width = continuous_src.cols;
        rga_src.height = continuous_src.rows;
        rga_src.wstride = static_cast<int>(continuous_src.step / continuous_src.elemSize());
        rga_src.hstride = continuous_src.rows;
        rga_src.format = RK_FORMAT_BGR_888;

        const im_rect source_rect{
            source_roi.x, source_roi.y, source_roi.width, source_roi.height};
        const im_rect destination_rect{destination_roi.x, destination_roi.y,
                                       destination_roi.width, destination_roi.height};
        IM_STATUS status = imcheck(rga_src, rga_canvas, source_rect, destination_rect);
        if (status == IM_STATUS_NOERROR) {
            status = improcessTask(job, rga_src, rga_canvas, {}, source_rect,
                                   destination_rect, {}, nullptr, IM_SYNC);
        }
        if (status != IM_STATUS_SUCCESS) {
            success = false;
            break;
        }
    }

    // 任何任务添加失败，取消整个批处理
    if (!success) {
        imcancelJob(job);        // 取消批处理，释放 GPU 资源
        return false;
    }

    // 4. 一次性提交所有任务给 RGA 硬件执行
    // IM_SYNC 表示同步模式：硬件执行完毕后才返回
    IM_STATUS status = imendJob(job, IM_SYNC, 0, nullptr);
    if (status != IM_STATUS_SUCCESS) {
        return false;
    }

    reportRgaResult(true); // 记录成功
    return true;
}

// ============================================================================
// updateTelemetry: 更新系统遥测数据（CPU、NPU、RGA、DDR 使用率，帧丢弃率等）
// @param snapshot 各帧源的当前快照
//
// 每秒采样一次，从 /proc/stat 读取 CPU 使用率，
// 从 /sys/kernel/debug/rknpu/load 读取 NPU 负载，
// 从 /sys/kernel/debug/rkrga/load 读取 RGA 负载，
// 从 /sys/class/devfreq/dmc/load 读取 DDR 负载。
// ============================================================================
void MosaicStreamPipeline::updateTelemetry(
    const std::vector<FrameHub::FrameSnapshot>& snapshot) {
    const auto now = std::chrono::steady_clock::now();
    // 每秒最多采样一次（限流）
    if (telemetry_.sampled_at.time_since_epoch().count() != 0 &&
        now - telemetry_.sampled_at < std::chrono::seconds(1)) return;

    // ---- 读取 CPU 使用率（/proc/stat） ----
    std::ifstream stat("/proc/stat");
    std::string cpu;
    uint64_t user = 0, nice = 0, system = 0, idle = 0, iowait = 0;
    uint64_t irq = 0, softirq = 0, steal = 0;
    stat >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
    const uint64_t idle_total = idle + iowait; // 总空闲时间
    const uint64_t total = user + nice + system + idle + iowait + irq + softirq + steal; // 总 CPU 时间
    if (telemetry_.previous_cpu_total > 0 && total > telemetry_.previous_cpu_total) {
        const uint64_t total_delta = total - telemetry_.previous_cpu_total; // 总时间增量
        const uint64_t idle_delta = idle_total - telemetry_.previous_cpu_idle; // 空闲时间增量
        // CPU 使用率 = (总时间 - 空闲时间) / 总时间 * 100
        telemetry_.cpu_percent = 100.0 * (total_delta - std::min(total_delta, idle_delta)) /
                                 std::max<uint64_t>(1, total_delta);
    }
    telemetry_.previous_cpu_total = total;    // 保存当前值用于下次差分
    telemetry_.previous_cpu_idle = idle_total;

    // ---- 读取帧丢弃率 ----
    uint64_t captured = 0;
    uint64_t dropped = media_service_
        ? media_service_->droppedFrames() + media_service_->droppedRtspPackets() : 0;
    for (const auto& item : snapshot) {
        captured += item.overlay.frames_captured; // 各源采集帧数
        dropped += item.overlay.frames_dropped;   // 各源丢弃帧数
    }
    if (telemetry_.previous_captured > 0 && captured >= telemetry_.previous_captured &&
        dropped >= telemetry_.previous_dropped) {
        const uint64_t captured_delta = captured - telemetry_.previous_captured; // 采集增量
        const uint64_t dropped_delta = dropped - telemetry_.previous_dropped;    // 丢弃增量
        telemetry_.drop_percent = captured_delta == 0 ? 0.0 :
            100.0 * std::min(dropped_delta, captured_delta) /
                static_cast<double>(captured_delta); // 丢弃率 = 丢弃 / 采集 * 100
    }
    telemetry_.previous_captured = captured;
    telemetry_.previous_dropped = dropped;

    // ---- 读取 NPU/RGA/DDR 负载 ----
    telemetry_.npu_percent = readFirstPercent("/sys/kernel/debug/rknpu/load");      // NPU 使用率
    telemetry_.rga_percent = readPercentAfter("/sys/kernel/debug/rkrga/load", "load ="); // RGA 使用率
    telemetry_.ddr_percent = readFirstPercent("/sys/class/devfreq/dmc/load");       // DDR 带宽使用率
    telemetry_.sampled_at = now; // 记录采样时间戳
}

// ============================================================================
// drawGlobalOverlay: 在画布顶部绘制全局遥测信息条
// 显示内容：输出 FPS、延迟、丢弃率、CPU/NPU/RGA/DDR 使用率
// @param canvas 目标画布（直接绘入）
// ============================================================================
void MosaicStreamPipeline::drawGlobalOverlay(cv::Mat& canvas) const {
    if (!config_.draw_overlay || canvas.empty()) return; // 不绘制叠加层或画布为空

    char text_buffer[256];
    const double encoded_fps = media_service_ ? media_service_->encodedFps() : 0.0;
    const double sent_fps = media_service_ ? media_service_->sentFps() : 0.0;
    const double output_fps = sent_fps > 0.0 ? sent_fps : encoded_fps;
    const int64_t latency_ms = media_service_ ? media_service_->lastFrameAgeMs() : 0; // 最后一帧延迟（ms）
    std::snprintf(text_buffer, sizeof(text_buffer),
        "OUT %.1f FPS   LAT %lld ms   DROP %.2f%%   CPU %.0f%%   NPU %.0f%%   RGA %.0f%%   DDR %.0f%%",
        output_fps, static_cast<long long>(latency_ms), telemetry_.drop_percent,
        telemetry_.cpu_percent, telemetry_.npu_percent,
        telemetry_.rga_percent, telemetry_.ddr_percent);

    // 自适应字体大小
    const double font_scale = std::clamp(canvas.cols / 2600.0, 0.48, 0.72);
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        text_buffer, cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);

    // 深色背景条 + 底部金色分割线
    const cv::Rect background(0, 0, canvas.cols, text_size.height + baseline + 18);
    cv::rectangle(canvas, background, {20, 22, 26}, cv::FILLED, cv::LINE_AA); // 深色背景
    cv::line(canvas, {0, background.height - 1},
             {canvas.cols, background.height - 1}, {190, 145, 70}, 1, cv::LINE_AA); // 金色分割线

    // 文字居中显示
    cv::putText(canvas, text_buffer,
        {(canvas.cols - text_size.width) / 2, text_size.height + 6},
        cv::FONT_HERSHEY_SIMPLEX, font_scale, {235, 238, 242}, 1, cv::LINE_AA);
}

/**
 * @brief 合成马赛克帧（绘制线程的核心方法）
 * @return 合成后的马赛克帧（canvas 的深拷贝，线程安全）
 *
 * 优化点：
 * 1. 预分配画布与 tile 缓冲，避免每帧 alloc（1280×720 BGR ≈ 2.7MB）
 * 2. 每次重绘两路最新帧，避免为维护脏块状态而整帧 copyTo
 * 3. RGA 硬件加速 resize（CPU 回退兜底）
 * 4. 三缓冲画布轮换，避免绘制与编码并发访问
 *
 * 布局逻辑：
 * - side_by_side 模式：左半（ExternalRtsp）+ 右半（Imx415）
 * - vertical 模式：上半（ExternalRtsp）+ 下半（Imx415）
 * - 单路源模式：仅有一路有数据时满屏显示
 */
cv::Mat MosaicStreamPipeline::composeFrame() {
    if (!hub_) {
        return {}; // FrameHub 未注入，返回空帧
    }

    // ---- 计算布局参数 ----
    const int rows = config_.input_mode == "vertical" ? 2 : 1;  // 垂直模式 2 行，水平模式 1 行
    const int cols = config_.input_mode == "vertical" ? 1 : 2;  // 垂直模式 1 列，水平模式 2 列
    const int tile_w = config_.width / cols;  // 每个 tile 的宽度
    const int tile_h = config_.height / rows; // 每个 tile 的高度

    // ---- 三缓冲画布轮换 ----
    // 三缓冲：写入当前画布后轮换，避免每帧重新分配
    cv::Mat& canvas = mosaic_canvas_[write_idx_.load(std::memory_order_relaxed)];

    // 预分配画布（仅首次或尺寸变化时分配）
    const bool canvas_busy = canvas.u != nullptr && canvas.u->refcount > 1;
    if (canvas.empty() || canvas.cols != config_.width ||
        canvas.rows != config_.height || canvas_busy) {
        // 创建画布（创建后默认内容不确定，先填充黑色底色，避免部分区域出现花屏）
        // busy 时必须替换 Mat header；create() 对同尺寸共享缓冲不会重新分配。
        canvas = cv::Mat(config_.height, config_.width, CV_8UC3);
    }
    // 如果某些 tile 没有新帧更新，保持旧内容不变，但新画布是空的，所以先填充黑色
    canvas.setTo(cv::Scalar(0, 0, 0)); // 黑色背景

    // ---- 延迟探测 RGA 硬件（仅一次） ----
    if (!rga_probed_) {
        probeRga(); // 探测 RGA 可用性
    }

    // ---- 双路布局：external_rtsp + imx415 ----
    const std::vector<FrameSource> sources{
        FrameSource::ExternalRtsp, // 源 0: 外部 RTSP（可见光）
        FrameSource::Imx415,       // 源 1: IMX415 摄像头（红外/热成像）
    };
    const std::array<int, 2> tile_source_map{{0, 1}}; // tile[0] → sources[0], tile[1] → sources[1]

    std::vector<FrameHub::FrameSnapshot> snapshot;

    // ---- 多源帧同步 ----
    // 同步回退：连续 10 次失败后跳过同步 300 帧，避免每帧 15ms 超时拖死帧率
    const bool should_try_sync = config_.sync_enabled && (sync_skip_cooldown_ <= 0);
    if (should_try_sync) {
        FrameHub::SyncResult pair;
        // 尝试获取同步帧对（时间戳差值在 sync_threshold_ms 以内）
        if (!hub_->takeSynchronized(sources[0], sources[1],
                static_cast<int64_t>(std::max(1, config_.sync_threshold_ms)) * 1000000LL, pair)) {
            // 同步失败：累加丢弃统计
            sync_drop_first_ += pair.dropped_first;
            sync_drop_second_ += pair.dropped_second;
            // 退化到非同步快照（独立获取各路最新帧）
            hub_->snapshot(sources, snapshot);
            // 连续失败计数：超过 10 次则进入冷却期
            if (++sync_fail_streak_ >= 10) {
                sync_skip_cooldown_ = 300;  // 冷却约 10 秒 @30fps，过后重试同步
                sync_fail_streak_ = 0;
            }
        } else {
            // 同步成功：使用同步帧对
            snapshot = {std::move(pair.first), std::move(pair.second)};
            ++sync_pairs_;
            sync_fail_streak_ = 0;      // 重置连续失败计数
            sync_skip_cooldown_ = 0;    // 退出冷却期
            sync_drop_first_ += pair.dropped_first;
            sync_drop_second_ += pair.dropped_second;
            // 统计同步时间差
            const int64_t absolute_delta = std::llabs(pair.delta_ns);
            sync_delta_total_ns_ += absolute_delta;                         // 累积差值
            sync_delta_max_ns_ = std::max(sync_delta_max_ns_, absolute_delta); // 最大差值
            // 每 300 对打印一次同步统计
            if (sync_pairs_ % 300 == 0) {
                std::cout << "[FrameSync] pairs=" << sync_pairs_
                          << " avg_ms=" << (sync_delta_total_ns_ / sync_pairs_ / 1000000.0)
                          << " max_ms=" << (sync_delta_max_ns_ / 1000000.0)
                          << " drops=" << sync_drop_first_ << ',' << sync_drop_second_ << '\n';
            }
        }
    } else {
        // 非同步模式（或冷却期内）：独立获取快照
        hub_->snapshot(sources, snapshot);
        if (sync_skip_cooldown_ > 0) --sync_skip_cooldown_; // 冷却倒计时
    }

    invalidateStaleSnapshots(snapshot, config_.source_stale_ms);

    // ---- 更新遥测数据 ----
    updateTelemetry(snapshot);

    // ---- 确定合成时间戳（取各路最小采集时间） ----
    last_composed_mono_ns_ = 0;
    last_composed_real_ms_ = 0;
    for (const auto& item : snapshot) {
        if (item.capture_mono_ns > 0) {
            last_composed_mono_ns_ = last_composed_mono_ns_ == 0
                ? item.capture_mono_ns
                : std::min(last_composed_mono_ns_, item.capture_mono_ns); // 取最早采集时间
        }
        last_composed_real_ms_ = std::max(last_composed_real_ms_,
            std::chrono::duration_cast<std::chrono::milliseconds>(
                item.timestamp.time_since_epoch()).count()); // 取最晚系统时间
    }

    // ---- 检测可用源数量 ----
    int available_count = 0; // 有数据的源计数
    int single_idx = -1;     // 单独源时的索引
    for (int i = 0; i < 2; ++i) {
        const int source_idx = tile_source_map[i];
        if (source_idx < 0 || source_idx >= static_cast<int>(snapshot.size())) {
            continue; // 索引越界，跳过
        }
        const auto& frame_ptr = snapshot[source_idx].frame;
        if (frame_ptr && !frame_ptr->empty()) {
            ++available_count;
            single_idx = i; // 记录最后一个有数据的源索引
        }
    }

    // ---- 单一路源模式 ----
    // 在 3568 仅启用可见光时，直接满屏输出，避免 2x2 黑块。
    if (available_count == 1 && single_idx >= 0) {
        const int source_idx = tile_source_map[single_idx];
        const cv::Mat& frame = *snapshot[source_idx].frame; // 获取帧数据
        const cv::Rect tile(0, 0, config_.width, config_.height); // 满屏 tile
        const TileGeometry geometry = tileGeometry(
            frame, tile, config_.preserve_aspect_ratio, config_.aspect_mode); // 计算几何映射

        // 使用 RGA 硬件加速进行 crop + resize + blit
        if (!rgaBlit(frame, canvas, geometry.source, geometry.destination)) {
            // RGA 失败，回退到 OpenCV CPU 路径
            cv::resize(frame(geometry.source), tile_bufs_[0], geometry.destination.size(),
                       0.0, 0.0, cv::INTER_LINEAR);
            tile_bufs_[0].copyTo(canvas(geometry.destination));
        }

        // 绘制检测框叠加层
        if (config_.draw_overlay) {
            drawTileOverlay(canvas, snapshot[source_idx].overlay,
                            geometry.destination, tile, geometry.source);
        }

        // 绘制全局遥测条
        drawGlobalOverlay(canvas);

        // 三缓冲轮换：切换到下一个画布
        const int next_write = (write_idx_.load(std::memory_order_relaxed) + 1) % 3;
        write_idx_.store(next_write, std::memory_order_relaxed);
        return canvas;
    }

    // ---- 多路源模式（标准 side_by_side / vertical） ----
    std::array<TileGeometry, 2> geometries{};
    std::array<bool, 2> tile_valid{{false, false}};
    std::vector<std::tuple<cv::Mat, cv::Rect, cv::Rect>> rga_tasks;
    rga_tasks.reserve(2);
    for (int i = 0; i < 2; ++i) {
        const int source_idx = tile_source_map[i];
        if (source_idx < 0 || source_idx >= static_cast<int>(snapshot.size())) {
            continue; // 索引无效
        }

        // FrameHub 保存无 OSD 的最新源帧；检测框在完成几何变换后统一绘制
        const auto& frame_ptr = snapshot[source_idx].frame;
        if (!frame_ptr || frame_ptr->empty()) {
            continue; // 帧为空，跳过此 tile
        }

        const cv::Mat& frame = *frame_ptr;

        // 计算 tile 在画布中的位置
        int r = i / cols; // 行索引
        int c = i % cols; // 列索引
        const cv::Rect tile(c * tile_w, r * tile_h, tile_w, tile_h); // tile 区域
        geometries[i] = tileGeometry(
            frame, tile, config_.preserve_aspect_ratio, config_.aspect_mode); // 计算几何映射
        tile_valid[i] = true;
        rga_tasks.emplace_back(frame, geometries[i].source, geometries[i].destination);
    }

    // 两路 crop/resize/blit 合并成一个 RGA job，避免逐 tile 同步等待和重复排队。
    const bool rga_composed = !rga_tasks.empty() && rgaBatchCompose(rga_tasks, canvas);
    for (int i = 0; i < 2; ++i) {
        if (!tile_valid[i]) continue;
        const int source_idx = tile_source_map[i];
        const cv::Mat& frame = *snapshot[source_idx].frame;
        const auto& geometry = geometries[i];
        const int r = i / cols;
        const int c = i % cols;
        const cv::Rect tile(c * tile_w, r * tile_h, tile_w, tile_h);

        if (!rga_composed &&
            !rgaBlit(frame, canvas, geometry.source, geometry.destination)) {
            // 旧版 BSP 不支持 task API 时退回单任务 RGA；只有单任务也失败才走 CPU。
            static thread_local uint64_t rga_fail_count = 0;
            if (++rga_fail_count % 100 == 1) {
                std::cerr << "[MosaicPipeline] RGA compose failed"
                          << " (count=" << rga_fail_count << "), CPU fallback\n";
            }
            cv::resize(frame(geometry.source), tile_bufs_[i], geometry.destination.size(),
                       0.0, 0.0, cv::INTER_AREA);
            tile_bufs_[i].copyTo(canvas(geometry.destination));
        }

        if (config_.draw_overlay) {
            drawTileOverlay(canvas, snapshot[source_idx].overlay,
                            geometry.destination, tile, geometry.source);
        }
    }

    // ---- 绘制 tile 间分割线 ----
    if (config_.input_mode == "side_by_side") {
        // 左右拼接：垂直分割线
        cv::line(canvas, {tile_w, 0}, {tile_w, config_.height},
                 {28, 31, 36}, 2, cv::LINE_AA);
    } else {
        // 上下拼接：水平分割线
        cv::line(canvas, {0, tile_h}, {config_.width, tile_h},
                 {28, 31, 36}, 2, cv::LINE_AA);
    }

    // ---- 绘制全局遥测条 ----
    drawGlobalOverlay(canvas);

    // ---- 三缓冲轮换 ----
    const int next_write = (write_idx_.load(std::memory_order_relaxed) + 1) % 3;
    write_idx_.store(next_write, std::memory_order_relaxed);
    return canvas;
}



/**
 * @brief 探测 RGA 硬件可用性（仅调用一次）
 *
 * 通过 querystring(RGA_VERSION) 查询 RGA 驱动版本判断硬件是否可用。
 * 结果缓存在 rga_available_ 和 rga_probed_ 中。
 */
void MosaicStreamPipeline::probeRga() {
    rga_probed_ = true;    // 标记已探测
    rga_available_ = false; // 默认不可用
    try {
        // 通过查询 RGA 版本来判断 RGA 是否可用
        const char* ver = querystring(RGA_VERSION);
        rga_available_ = (ver != nullptr && ver[0] != '\0'); // 非空版本号表示可用
        if (rga_available_) {
            std::cout << "MosaicStreamPipeline: RGA hardware available, version: "
                      << ver << std::endl;
        }
    } catch (...) {
        rga_available_ = false; // 异常 → 不可用
    }
    if (!rga_available_) {
        std::cout << "MosaicStreamPipeline: RGA not available, using CPU resize"
                  << std::endl;
    }
}

/**
 * @brief RGA 硬件加速 crop + resize，并直接写入完整画布 ROI
 * @param src            源帧（BGR888 格式）
 * @param canvas         目标画布（直接写入）
 * @param requested_roi  请求的源帧裁剪区域
 * @param destination_roi 目标画布中的绘制区域
 * @return               true 表示 RGA 操作成功
 *
 * 使用 improcess（RGA 一站式 API）一次完成 crop + resize + 颜色空间转换。
 * 此函数对输入数据做了严格的前置校验（尺寸、指针、连续性），防止内核 Oops。
 */
bool MosaicStreamPipeline::rgaBlit(const cv::Mat& src, cv::Mat& canvas,
                                   const cv::Rect& requested_roi,
                                   const cv::Rect& destination_roi) {
    // 前置检查：RGA 可用、帧有效、格式正确
    if (!rga_available_ || !rgaRuntimeHealthy() || src.empty() || canvas.empty() ||
        src.type() != CV_8UC3 || canvas.type() != CV_8UC3) {
        return false;
    }
    RgaSubmissionGuard rga_guard; // RGA 提交守卫
    if (!rgaRuntimeHealthy()) return false;

    // 尺寸校验：拒绝明显损坏的帧（超过 4K 分辨率的帧 RGA 无法处理）
    if (src.cols <= 0 || src.rows <= 0 || src.cols > 4096 || src.rows > 4096) {
        return false;
    }
    // 数据指针校验：RGA 对空指针会直接内核 Oops
    if (src.data == nullptr || src.datastart == nullptr) {
        return false;
    }

    // ROI 边界裁剪：确保请求区域在源帧范围内
    const cv::Rect full(0, 0, src.cols, src.rows);
    const cv::Rect roi = requested_roi.area() > 0 ? (requested_roi & full) : full;
    if (roi.width <= 0 || roi.height <= 0 || roi.width > 4096 || roi.height > 4096) return false;
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > src.cols || roi.y + roi.height > src.rows) return false;

    // 目标区域校验
    const cv::Rect canvas_bounds(0, 0, canvas.cols, canvas.rows);
    const cv::Rect destination = destination_roi & canvas_bounds;
    if (destination != destination_roi || destination.width <= 0 || destination.height <= 0 ||
        !canvas.isContinuous() || canvas.data != canvas.datastart) return false;

    // 子矩阵的 data 可能已经偏离底层分配基址；复制成独立连续图像后再提交。
    // 这样 RGA 获取到的地址、尺寸和可访问范围始终一致。
    const cv::Mat safe_src = (!src.isContinuous() || src.data != src.datastart)
                                 ? src.clone()
                                 : src;

    // 始终把图像基址交给 RGA，裁剪区域通过 srect 描述。
    // 不能把 vir_addr 偏移到 ROI 后仍声明整图 stride/height，否则驱动导入范围会越界。

    // ---- 配置 RGA 源缓冲描述 ----
    rga_buffer_t rga_src{};
    rga_src.vir_addr = const_cast<uint8_t*>(safe_src.data);            // 源数据虚拟地址
    rga_src.width = safe_src.cols;                                     // 源帧宽度
    rga_src.height = safe_src.rows;                                    // 源帧高度
    rga_src.wstride = static_cast<int>(safe_src.step / safe_src.elemSize()); // 行步长（像素单位）
    rga_src.hstride = safe_src.rows;                                   // 列步长
    rga_src.format = RK_FORMAT_BGR_888;                                // BGR 8-8-8

    // ---- 配置 RGA 目标缓冲描述 ----
    // 配置完整画布；目标位置只通过 destination_rect 描述
    rga_buffer_t rga_dst{};
    rga_dst.vir_addr = canvas.data;                                    // 画布数据地址
    rga_dst.width = canvas.cols;                                       // 画布宽度
    rga_dst.height = canvas.rows;                                      // 画布高度
    rga_dst.wstride = static_cast<int>(canvas.step / canvas.elemSize()); // 行步长
    rga_dst.hstride = canvas.rows;                                     // 列步长
    rga_dst.format = RK_FORMAT_BGR_888;                                // BGR 8-8-8

    // ---- 定义源 ROI 和目标 ROI ----
    const im_rect source_rect{roi.x, roi.y, roi.width, roi.height};
    const im_rect destination_rect{
        destination.x, destination.y, destination.width, destination.height};

    // ---- 步骤 1: 参数检查 ----
    IM_STATUS status = imcheck(rga_src, rga_dst, source_rect, destination_rect);
    if (status != IM_STATUS_NOERROR) {
        reportRgaResult(false);
        return false;
    }

    // ---- 步骤 2: 执行 RGA 处理 ----
    // improcess 一站完成 crop(source_rect) + resize + blit(destination_rect)
    // IM_SYNC: 同步执行，函数返回时硬件已完成操作
    status = improcess(rga_src, rga_dst, {}, source_rect, destination_rect, {}, IM_SYNC);
    if (status != IM_STATUS_SUCCESS) {
        reportRgaResult(false);
        return false;
    }

    reportRgaResult(true);
    return true;
}

} // namespace pipeline
