#pragma once

/**
 * @file streaming_manager.h
 * @brief 流管理器头文件
 * 
 * 该文件定义了StreamingManager类，负责管理视频流的推送，
 * 支持RTSP流推送，包括视频编码与发送。
 */

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <string>
#include <chrono>
#include <condition_variable>

/**
 * @struct StreamingData
 * @brief 流数据结构体
 * 
 * 存储流推送所需的数据，包括视频帧和时间戳。
 */
struct StreamingData {
    int stream_id;                           ///< 流ID
    cv::Mat frame;                           ///< 视频帧（BGR格式）
    std::chrono::system_clock::time_point timestamp; ///< 时间戳
};

/**
 * @struct StreamingConfig
 * @brief 流配置结构体
 * 
 * 存储流推送的配置参数，包括推流地址、分辨率、帧率等。
 */
struct StreamingConfig {
    std::string rtsp_url;           ///< RTSP推流地址
    int width = 1280;              ///< 推流分辨率宽度
    int height = 720;               ///< 推流分辨率高度
    int fps = 25;                   ///< 推流帧率
    int bitrate = 2000000;          ///< 推流码率
    bool enable_rtsp = true;        ///< 是否启用RTSP推流
    int max_queue_size = 3;         ///< 推流队列最大深度
    bool drop_oldest_when_full = true; ///< 队列满时是否丢弃最旧帧
    bool consume_latest_only = true; ///< 消费端是否每轮只保留最新帧
    bool resize_on_mismatch = false; ///< 分辨率不匹配时是否自动 resize
};

/**
 * @class StreamingManager
 * @brief 流管理器类
 * 
 * 负责管理视频流的推送，支持RTSP流推送，包括视频编码与发送。
 */
class StreamingManager {
public:
    /**
     * @brief 构造函数，初始化流管理器
     * 
     * 初始化流管理器的成员变量，设置默认值。
     */
    StreamingManager();
    
    /**
     * @brief 析构函数，停止流管理器
     * 
     * 停止流推送，释放相关资源。
     */
    ~StreamingManager();
    
    /**
     * @brief 初始化流管理器
     * @param config 流配置参数
     * @return 是否初始化成功
     * 
     * 根据配置初始化流管理器，设置流参数，初始化RTSP流。
     */
    bool initialize(const StreamingConfig& config);
    
    /**
     * @brief 添加流数据到队列
     * @param data 流数据
     * 
     * 将流数据添加到队列，用于后续的流推送。
     */
    void addStreamingData(StreamingData data);

    /**
     * @brief 启动流推送
     * 
     * 启动流推送线程，开始推送视频流。
     */
    void startStreaming();
    
    /**
     * @brief 停止流推送
     * 
     * 停止流推送线程，释放相关资源。
     */
    void stopStreaming();
    
    /**
     * @brief 检查流推送状态
     * @return 是否正在推送流
     * 
     * 检查流推送是否处于活跃状态。
     */
    bool isStreaming() const { return streaming_active_.load(); }
    
    /**
     * @struct StreamingStats
     * @brief 流统计信息结构体
     * 
     * 存储流推送的统计信息，如发送的帧数、丢弃的帧数、帧率等。
     */
    struct StreamingStats {
        int frames_sent = 0;                ///< 发送的帧数
        int frames_dropped = 0;             ///< 丢弃的帧数
        double fps = 0.0;                   ///< 帧率
        std::chrono::system_clock::time_point last_frame_time; ///< 最后一帧的时间
    };
    
    /**
     * @brief 获取流统计信息
     * @return 流统计信息
     * 
     * 获取流推送的统计信息，包括发送的帧数、丢弃的帧数和帧率等。
     */
    StreamingStats getStats() const;

private:
    /**
     * @brief 流推送工作线程
     * 
     * 从队列中获取流数据，进行编码和推送。
     */
    void streamingWorker();
    
    /**
     * @brief 初始化RTSP流
     * @return 是否初始化成功
     * 
     * 初始化RTSP流，设置编码器和格式上下文。
     */
    bool initializeRTSP();
    
    /**
     * @brief 发送RTSP帧
     * @param frame 帧数据
     * @return 是否发送成功
     * 
     * 发送帧数据到RTSP流。
     */
    bool sendRTSPFrame(const cv::Mat& frame);

private:
    static constexpr int kMaxQueueSize = 3; ///< 默认推流队列最大深度

    StreamingConfig config_;           ///< 流配置参数
    std::atomic<bool> streaming_active_; ///< 流推送是否活跃
    std::atomic<bool> should_stop_;     ///< 是否应该停止流推送
    
    // 流推送队列
    std::queue<StreamingData> streaming_queue_; ///< 流数据队列
    std::mutex queue_mutex_;           ///< 队列互斥锁
    std::condition_variable queue_cv_; ///< 队列条件变量
    
    // 流推送线程
    std::thread streaming_thread_;     ///< 流推送线程
    

    // FFmpeg相关（RTSP）
    void* avformat_context_rtsp_;      ///< FFmpeg格式上下文（RTSP）
    void* avcodec_context_rtsp_;       ///< FFmpeg编解码器上下文（RTSP）
    void* sws_context_cache_;          ///< SwsContext 缓存（复用，避免每帧分配）
    void* av_frame_cache_;             ///< AVFrame 缓存（复用，避免每帧分配）
    void* av_packet_cache_;            ///< AVPacket 缓存（复用，避免每帧分配）
    void* mpp_encoder_;                ///< 原生 MPP H.264 编码器
    
    // 统计信息
    mutable std::mutex stats_mutex_;    ///< 统计信息互斥锁
    StreamingStats stats_;             ///< 流推送统计信息
    
    // 帧率计算
    std::chrono::system_clock::time_point last_fps_time_; ///< 最后计算帧率的时间
    int frame_count_;                  ///< 帧计数
    int64_t rtsp_frame_index_ = 0;     ///< RTSP帧序号

    /// 推流重连相关
    static constexpr int kMaxConsecutiveFailures = 30;  ///< 连续失败阈值触发重连
    int consecutive_send_failures_ = 0;
    bool tryReconnectRTSP();
};
