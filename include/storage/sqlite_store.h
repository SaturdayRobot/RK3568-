#pragma once

#include <atomic>      // 原子变量，用于运行时统计的无锁计数
#include <cstdint>     // 定宽整数类型（int64_t, uint64_t 等）
#include <mutex>       // 互斥锁，保护数据库句柄和写缓冲区
#include <string>      // 标准字符串
#include <vector>      // 动态数组，用于批量写入缓冲和待上传 ID 列表

#include "config/ini_config.h"  // INI 配置文件解析器

// P2-3 修复：前置声明 sqlite3，避免使用 void* 的类型不安全
// 实际定义在 <sqlite3.h> 中，仅在 .cpp 中引入，减少头文件依赖
struct sqlite3;

namespace data_lifecycle {

// ============================================================================
// StorageConfig 结构体 —— 本地 SQLite 存储的完整配置
//
// 该结构体覆盖了数据库文件的路径管理、写入批量策略、数据保留策略、
// 以及上传回写时的行为控制。所有字段均可从 INI 配置文件加载。
// ============================================================================
struct StorageConfig {
    // --- 基础开关与路径 ---
    bool enable = false;                               // 是否启用本地存储（关闭后所有写操作直接返回 false）
    std::string base_dir = "../data/sqlite";           // 数据库存放目录（绝对路径或相对于工作目录）
    std::string db_prefix = "edge_data_";              // 数据库文件名前缀，最终文件名为 <prefix><YYYYMMDD>.db
    int retention_days = 3;                            // 数据保留天数，超期的 .db 文件会被自动清理
    int max_batch = 200;                               // 单次 fetchPending 拉取待上传记录的最大数量
    bool wal = true;                                   // 是否启用 WAL（Write-Ahead Logging）模式提高并发写入能力
    std::string sync = "NORMAL";                       // SQLite 同步级别：OFF(最快/可能损坏) / NORMAL(平衡) / FULL(最安全)

    // --- 批量写入策略（用于提升本地入库吞吐） ---
    int write_batch_size = 50;                         // 触发批量写入的最小缓冲条数（达到此值强制刷盘）
    int write_flush_ms = 1000;                         // 批量写入的最大等待间隔（毫秒），超时也触发刷盘
    bool flush_before_fetch = false;                   // 拉取待上传记录前是否强制将缓冲刷入数据库
    bool flush_before_mark = false;                    // 回写上传状态（markUploaded/markFailed/markDead）前是否强制刷盘
    int mark_update_chunk_size = 64;                   // 批量回写上传状态时的分块大小（每次 UPDATE 处理的 ID 数量）

    // --- 推理统计 ---
    int inference_sample_ms = 5000;                    // 推理统计的采样间隔（毫秒），控制统计数据的产生频率
};

// ============================================================================
// InferenceStats 结构体 —— 单帧推理统计信息
//
// 记录每一帧推理的产出（各类检测对象的数量）和推理执行状态。
// 会被序列化为 JSON 格式后存入 upload_queue 表。
// ============================================================================
struct InferenceStats {
    int stream_id = 0;               // 视频流 ID，区分多路输入
    int person_count = 0;            // 本帧检测到的人员数量
    int cat_count = 0;               // 本帧检测到的猫数量
    int dog_count = 0;               // 本帧检测到的狗数量
    int fire_count = 0;              // 本帧检测到的火焰数量
    int smoke_count = 0;             // 本帧检测到的烟雾数量
    int ppe_count = 0;               // 本帧检测到的安全防护装备数量
    bool infer_executed = true;      // 本帧是否实际执行了推理（true=正常推理）
    bool infer_skipped = false;      // 本帧是否跳过了推理（true=因某种原因跳过）
    bool detection_reused = false;   // 本帧是否复用了上一帧的检测结果（节省推理资源）
    std::string infer_skip_reason;   // 推理跳过原因（如 "thermal"/"interval"/"service_not_ready"/"infer_failed"）
    std::int64_t ts_ms = 0;          // 事件时间戳（毫秒），使用真实系统时间
    std::int64_t capture_mono_ns = 0; // 触发推理的采集帧单调时间（纳秒），不受 NTP 校时影响，用于事件录像边界计算
};

// ============================================================================
// UploadRecord 结构体 —— 从本地数据库拉取的待上传记录
//
// 每条记录对应 upload_queue 表中的一行，包含事件类型和 JSON 负载。
// 支持重试计数和最后一次尝试时间，用于上传失败后的重试控制。
// ============================================================================
struct UploadRecord {
    std::int64_t id = 0;             // 主键 ID（upload_queue 表的自增主键）
    std::int64_t ts_ms = 0;          // 原始事件时间戳（毫秒），用于服务端排序和去重
    std::string type;                // 记录类型标识（如 "sensor"/"inference"/"device_status"/"log"/"frame_meta"）
    std::string payload;             // JSON 格式的实际数据负载
    int retry_count = 0;             // 已重试次数（每次上传失败后递增）
    std::int64_t last_try_ts = 0;    // 最近一次重试的时间戳（毫秒），用于退避策略计算
};

// ============================================================================
// SqliteStore 类 —— SQLite 本地存储管理器
//
// 核心职责：
//   1. 按天分库：每天自动创建独立的 .db 文件，便于管理和清理
//   2. 批量写入优化：使用写缓冲区 + 批量事务，减少 SQLite 写入次数，提升吞吐
//   3. 上传队列管理：通过 upload_queue 表维护待上传/已上传/失败/死亡状态的生命周期
//   4. 自动清理：按 retention_days 配置自动删除过期数据库文件
//   5. 并发安全：双锁设计（db_mutex_ 保护数据库操作，write_buf_mutex_ 保护写缓冲）
//
// 线程模型：
//   - 传感器/推理线程：调用 insertXxx 系列方法写入数据（仅获取写缓冲锁，低延迟）
//   - 上传线程：调用 fetchPending/markXxx 系列方法读取和回写状态（获取数据库锁）
//   - 双锁分离确保高频写入不会被低频上传操作阻塞
// ============================================================================
class SqliteStore {
public:
    SqliteStore();   // 默认构造函数，初始化所有成员为默认值
    ~SqliteStore();  // 析构函数：自动关闭数据库连接并释放资源

    // 从 INI 配置文件加载存储配置（静态方法，可在构造前使用）
    // 参数 path: INI 文件的路径
    // 参数 out: 输出参数，填充解析后的 StorageConfig
    // 返回值: true=加载成功，false=文件不存在或解析失败
    static bool loadFromIni(const std::string& path, StorageConfig& out);

    // 初始化存储模块：创建目录、打开当天的数据库、建表、清理旧文件
    // 参数 config: 存储配置（通常由 loadFromIni 获得）
    // 返回值: true=初始化成功（或未启用），false=初始化失败
    bool initialize(const StorageConfig& config);
    // 关闭数据库连接：强制刷盘、关闭 SQLite 句柄
    void close();

    // 插入推理统计记录（序列化为 JSON 后存入 upload_queue）
    // 参数 stats: 推理统计数据
    // 返回值: true=成功写入缓冲，false=失败
    bool insertInferenceStats(const InferenceStats& stats);
    // 插入通用原始记录（直接传入类型、时间戳和 JSON 负载）
    // 参数 type: 记录类型（如 "sensor"/"log" 等）
    // 参数 ts_ms: 事件时间戳（毫秒）
    // 参数 payload: JSON 格式的负载数据
    // 返回值: true=成功写入缓冲，false=失败
    bool insertRawRecord(const std::string& type,
                         std::int64_t ts_ms,
                         const std::string& payload);

    // 拉取待上传记录列表（uploaded=0），按时间戳升序排列
    // 参数 limit: 最大返回条数
    // 参数 out: 输出参数，填充 UploadRecord 列表
    // 返回值: true=拉取到至少一条记录，false=无记录或出错
    bool fetchPending(int limit, std::vector<UploadRecord>& out);

    // 将指定 ID 的记录标记为"已上传"（uploaded=1）
    // 使用分批 UPDATE + 事务确保大数据量时的性能和一致性
    // 参数 ids: 要标记为已上传的记录 ID 列表
    // 返回值: true=标记成功，false=失败（事务已回滚）
    bool markUploaded(const std::vector<std::int64_t>& ids);
    // 将指定 ID 的记录标记为"上传失败"（retry_count+1，记录错误信息和尝试时间）
    // 参数 ids: 要标记失败的记录 ID 列表
    // 参数 error: 失败原因描述
    // 参数 ts_ms: 当前尝试时间戳（毫秒）
    // 返回值: true=标记成功，false=失败
    bool markFailed(const std::vector<std::int64_t>& ids,
                    const std::string& error,
                    std::int64_t ts_ms);
    // 将指定 ID 的记录标记为"死亡"（uploaded=2，永久放弃上传）
    // 用于超过最大重试次数或数据本身有问题的记录
    // 参数 ids: 要标记为死亡的记录 ID 列表
    // 参数 error: 标记死亡的原因
    // 返回值: true=标记成功，false=失败
    bool markDead(const std::vector<std::int64_t>& ids,
                  const std::string& error);

    // 运行时统计信息（用于监控上报）
    struct RuntimeStats {
        std::int64_t last_flush_us = 0;    // 最近一次批量写入耗时（微秒）
        std::uint64_t flush_count = 0;     // 累计批量写入次数
        std::uint64_t mark_count = 0;      // 累计标记操作条数
    };

    // 查询当前未上传记录的数量
    // 返回值: 未上传记录数（uploaded=0）
    int pendingUploadCount();
    // 获取运行时统计信息（线程安全读取原子变量）
    RuntimeStats runtimeStats() const;

    // 返回配置的最大单次拉取条数（供上传模块参考）
    int maxBatch() const { return config_.max_batch; }
    // 返回存储是否启用（供外部快速判断）
    bool enabled() const { return config_.enable; }

private:
    // PendingWrite 结构体 —— 批量缓冲写入项
    // 暂存于内存中的待写入记录，积累到一定量后批量写入 SQLite
    struct PendingWrite {
        std::string type;        // 记录类型标识
        std::int64_t ts_ms = 0;  // 时间戳（毫秒）
        std::string payload;     // JSON 负载数据
    };

    // --- 内部方法 ---

    // 根据时间戳打开对应的数据库（按天切库）
    // 如果日期与当前已打开的库不同，则自动关闭旧库并打开新库
    // 参数 ts_ms: 时间戳（毫秒），用于确定数据库日期
    // 返回值: true=数据库已就绪，false=打开失败
    bool openForTimestamp(std::int64_t ts_ms);
    // 创建业务所需的表结构（upload_queue 表及其索引）
    // 同时兼容历史库：尝试 ALTER TABLE 补充缺失列
    // 返回值: true=表已就绪，false=创建失败
    bool createTables();
    // 清理超过保留天数的历史数据库文件
    // 遍历 base_dir 目录，按文件名中的日期前缀判断并删除过期 .db 文件
    void cleanupOldFiles();
    // 根据日期字符串构造数据库文件的完整路径
    // 参数 day: 日期字符串（格式 YYYYMMDD）
    // 返回值: 完整文件路径（如 "../data/sqlite/edge_data_20260105.db"）
    std::string buildDbPath(const std::string& day) const;
    // 将毫秒时间戳转换为日期字符串（YYYYMMDD 格式）
    // 参数 ts_ms: 时间戳（毫秒）
    // 返回值: 8 位日期字符串
    std::string toDayString(std::int64_t ts_ms) const;
    // 获取当前系统时间的毫秒值（封装 toUnixMs 调用）
    std::int64_t nowMs() const;

    // 插入单条通用记录的内部实现
    // 参数 type: 记录类型
    // 参数 ts_ms: 事件时间戳（毫秒）
    // 参数 payload: JSON 负载数据
    // 返回值: true=已写入缓冲或数据库，false=失败
    bool insertRecord(const std::string& type, std::int64_t ts_ms, const std::string& payload);
    // 刷新批量写缓冲区到 SQLite 数据库
    // 注意：调用前必须已持有 db_mutex_（数据库互斥锁）
    // 参数 force: true=强制刷盘（忽略批量大小和时间阈值）, false=仅在达到阈值时刷盘
    // 返回值: true=刷盘成功（或缓冲区为空），false=写入失败
    bool flushPendingLocked(bool force);

    // 基础 JSON 字符串转义工具：对双引号、反斜杠、换行等特殊字符进行转义
    // 参数 input: 原始字符串
    // 返回值: 转义后的安全 JSON 字符串
    std::string escapeJson(const std::string& input) const;

    // --- 成员变量 ---

    StorageConfig config_;               // 当前生效的存储配置
    sqlite3* db_ = nullptr;              // SQLite 数据库连接句柄（P2-3 修复：使用前置声明类型，替代 void*）
    std::string current_day_;            // 当前已打开数据库对应的日期（YYYYMMDD 格式），用于判断是否需要切库
    std::mutex db_mutex_;                // 数据库互斥锁：保护 db_ 句柄和所有 SQLite API 调用（fetch/mark/close）
    std::mutex write_buf_mutex_;         // 写缓冲互斥锁：保护 pending_writes_（与 db_mutex_ 分离，避免传感器写入被上传阻塞）

    std::vector<PendingWrite> pending_writes_; // 批量写缓冲区：暂存待写入的推理统计和原始记录
    std::int64_t last_flush_ms_ = 0;          // 最近一次刷盘的时间（毫秒），用于判断时间阈值

    std::atomic<std::int64_t> last_flush_us_{0};     // 最近一次刷盘耗时（微秒），原子变量，线程安全读取
    std::atomic<std::uint64_t> flush_count_{0};      // 累计刷盘次数，原子变量
    std::atomic<std::uint64_t> mark_count_{0};       // 累计标记操作条数，原子变量
};

} // namespace data_lifecycle
