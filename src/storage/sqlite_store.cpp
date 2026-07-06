#include "storage/sqlite_store.h"

#include <algorithm>    // std::min, std::max
#include <chrono>       // 系统时钟和高精度计时
#include <ctime>        // localtime_r 线程安全时间转换
#include <filesystem>   // 目录遍历、文件清理操作
#include <iomanip>      // 流格式控制
#include <iostream>     // 控制台错误输出
#include <sstream>      // 字符串流（路径拼接、JSON 构建）

#ifdef HAVE_SQLITE3
#include <sqlite3.h>    // SQLite3 C API（条件编译，确保依赖可用时才引入）
#endif

namespace data_lifecycle {

namespace {
// 将 system_clock::time_point 转换为 Unix 毫秒时间戳
// 统一将时间点转换为毫秒值，便于写库与时间比较
// 参数 tp: C++ chrono 时间点
// 返回值: 毫秒级 Unix 时间戳
std::int64_t toUnixMs(std::chrono::system_clock::time_point tp) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count();
}

// 构建 SQL 占位符字符串（?,?,? ...），用于 IN 子句的批量绑定
// 参数 count: 占位符数量
// 返回值: 由 count 个 "?" 以逗号连接的字符串，如 count=3 返回 "?,?,?"
std::string buildPlaceholders(std::size_t count) {
    if (count == 0) {
        return ""; // 空集，返回空字符串（调用方应确保不传 0）
    }

    std::string out;
    out.reserve(count * 2); // 预分配空间：每个 ? 占 1 字符 + 1 逗号 ≈ 2 字符
    for (std::size_t i = 0; i < count; ++i) {
        if (i > 0) {
            out += ','; // 非首项时添加逗号分隔
        }
        out += '?'; // 添加占位符
    }
    return out;
}
} // namespace

// ============================================================================
// SqliteStore 类实现
// ============================================================================

// 默认构造函数：所有成员初始化为默认值
SqliteStore::SqliteStore() = default;

// 析构函数：确保数据库连接正确关闭
SqliteStore::~SqliteStore() {
    close(); // 关闭前自动强制刷盘并释放 SQLite 句柄
}

// 从 INI 配置文件加载存储模块的所有配置项（静态方法，可在构造前使用）
// 参数 path: INI 文件路径
// 参数 out: 输出参数，填充解析结果
// 返回值: true=加载成功，false=文件打开失败
bool SqliteStore::loadFromIni(const std::string& path, StorageConfig& out) {
    IniConfig cfg;
    if (!cfg.load(path)) {
        return false; // INI 文件不存在或格式错误
    }

    // 逐个读取配置项，使用 getBool/getString/getInt 方法，未配置时使用默认值

    // 基础开关与目录配置
    out.enable = cfg.getBool("storage", "enable", false);             // 存储启用开关
    out.base_dir = cfg.getString("storage", "base_dir", "../data/sqlite"); // 数据库目录
    out.db_prefix = cfg.getString("storage", "db_prefix", "edge_data_");   // 文件名前缀
    out.retention_days = cfg.getInt("storage", "retention_days", 3);      // 保留天数
    out.max_batch = cfg.getInt("storage", "max_batch", 200);              // 最大拉取条数
    out.wal = cfg.getBool("storage", "wal", true);                        // WAL 模式
    out.sync = cfg.getString("storage", "sync", "NORMAL");                // 同步级别

    // 写入聚合与刷新策略配置
    out.write_batch_size = cfg.getInt("storage", "write_batch_size", 50);      // 批量大小阈值
    out.write_flush_ms = cfg.getInt("storage", "write_flush_ms", 1000);        // 刷新间隔
    out.flush_before_fetch = cfg.getBool("storage", "flush_before_fetch", false); // 拉取前刷盘
    out.flush_before_mark = cfg.getBool("storage", "flush_before_mark", false);   // 标记前刷盘
    out.mark_update_chunk_size = cfg.getInt("storage", "mark_update_chunk_size", 64); // 回写分块大小
    if (out.mark_update_chunk_size <= 0) {
        out.mark_update_chunk_size = 64; // 防御性保护：分块大小必须为正数，避免无限循环
    }

    // 推理统计采样间隔
    out.inference_sample_ms = cfg.getInt("storage", "inference_sample_ms", 5000);
    return true;
}

// 初始化存储模块：创建目录、打开当天数据库、建表、清理旧文件
// 参数 config: 存储配置
// 返回值: true=初始化成功，false=失败
bool SqliteStore::initialize(const StorageConfig& config) {
    config_ = config; // 缓存配置供后续使用
    if (!config_.enable) {
        // 未启用存储时也返回成功，便于上层统一流程（外部调用方无需检查 enable 状态）
        return true;
    }

#ifndef HAVE_SQLITE3
    // 编译时 SQLite3 不可用（HAVE_SQLITE3 宏未定义），输出提示并返回失败
    std::cerr << "SqliteStore: sqlite3 not available, disabled" << std::endl;
    return false;
#else
    try {
        // 创建存储目录（如已存在则 create_directories 无实际操作）
        std::filesystem::create_directories(config_.base_dir);
    } catch (...) {
        std::cerr << "SqliteStore: create base_dir failed: " << config_.base_dir << std::endl;
        return false;
    }

    // 打开当天数据库并初始化状态
    std::lock_guard<std::mutex> dlock(db_mutex_);
    if (!openForTimestamp(nowMs())) {
        return false; // 数据库打开失败
    }

    // 清理内存中的待写队列（避免上次异常退出残留数据被重复写入）并记录刷新时刻
    {
        std::lock_guard<std::mutex> wlock(write_buf_mutex_);
        pending_writes_.clear();          // 清空写缓冲区
        last_flush_ms_ = nowMs();         // 重置上次刷盘时间
    }
    // 按保留天数清理过期的旧数据库文件
    cleanupOldFiles();
    return true;
#endif
}

// 关闭数据库连接：强制刷盘、执行 WAL 检查点、关闭 SQLite 句柄
void SqliteStore::close() {
#ifdef HAVE_SQLITE3
    std::lock_guard<std::mutex> lock(db_mutex_);
    // 强制刷新所有未写入的缓冲数据（force=true 忽略阈值条件）
    flushPendingLocked(true);
    if (db_) {
        // 关闭数据库连接
        // P2-3 修复：db_ 已改为 sqlite3* 类型，无需强制转换
        sqlite3_close(db_); // 关闭 SQLite 数据库句柄
        db_ = nullptr;      // 防止悬空指针
    }
#endif
}

// 插入推理统计记录：将 InferenceStats 序列化为 JSON 后存入 upload_queue
// 参数 stats: 推理统计数据结构体（包含各检测计数和推理执行状态）
// 返回值: true=成功写入缓冲，false=失败或存储未启用
bool SqliteStore::insertInferenceStats(const InferenceStats& stats) {
    if (!config_.enable) {
        return false; // 存储未启用
    }

    // 手动构建紧凑的 JSON 字符串（避免引入第三方 JSON 库依赖）
    std::ostringstream oss;
    oss << "{";
    oss << "\"stream_id\":" << stats.stream_id << ",";          // 视频流 ID
    oss << "\"person_count\":" << stats.person_count << ",";    // 人员检测数
    oss << "\"cat_count\":" << stats.cat_count << ",";          // 猫检测数
    oss << "\"dog_count\":" << stats.dog_count << ",";          // 狗检测数
    oss << "\"fire_count\":" << stats.fire_count << ",";        // 火焰检测数
    oss << "\"smoke_count\":" << stats.smoke_count << ",";      // 烟雾检测数
    oss << "\"ppe_count\":" << stats.ppe_count << ",";          // PPE 检测数
    oss << "\"infer_executed\":" << (stats.infer_executed ? 1 : 0) << ","; // 是否执行推理（布尔转 0/1）
    oss << "\"infer_skipped\":" << (stats.infer_skipped ? 1 : 0) << ",";   // 是否跳过推理
    oss << "\"detection_reused\":" << (stats.detection_reused ? 1 : 0) << ","; // 是否复用检测结果
    oss << "\"infer_skip_reason\":\"" << escapeJson(stats.infer_skip_reason) << "\""; // 跳过原因（需转义）
    oss << "}";

    // 以 "inference" 类型写入 insertRecord
    return insertRecord("inference", stats.ts_ms, oss.str());
}

// 插入通用原始记录（外部调用方直接提供类型、时间戳和 JSON 负载）
// 参数 type: 记录类型标识
// 参数 ts_ms: 事件时间戳（毫秒）
// 参数 payload: 已序列化的 JSON 负载字符串
// 返回值: true=成功写入，false=失败
bool SqliteStore::insertRawRecord(const std::string& type,
                                  std::int64_t ts_ms,
                                  const std::string& payload) {
    if (!config_.enable) {
        return false; // 存储未启用
    }
    if (type.empty() || payload.empty()) {
        return false; // 类型或负载为空时拒绝写入，避免产生无效记录
    }
    return insertRecord(type, ts_ms, payload);
}

// 拉取待上传记录列表（uploaded=0），按时间戳升序排列，取前 limit 条
// 参数 limit: 最大返回条数（通常由 config_.max_batch 控制）
// 参数 out: 输出参数，填充 UploadRecord 列表
// 返回值: true=至少拉取到一条记录，false=无记录或出错
bool SqliteStore::fetchPending(int limit, std::vector<UploadRecord>& out) {
    out.clear(); // 先清空输出列表
    if (!config_.enable) {
        return false; // 存储未启用
    }

#ifndef HAVE_SQLITE3
    return false; // SQLite3 不可用
#else
    std::lock_guard<std::mutex> lock(db_mutex_); // 获取数据库锁
    // 可配置：拉取前是否强制刷盘（减少开关可以降低 I/O 抖动）
    flushPendingLocked(config_.flush_before_fetch);
    if (!db_) {
        return false; // 数据库未打开
    }

    // SQL：按 ts_ms 升序选择前 N 条未上传记录（FIFO 策略，最早产生的先上传）
    const char* sql =
        "SELECT id, ts_ms, type, payload, retry_count, last_try_ts FROM upload_queue "
        "WHERE uploaded=0 ORDER BY ts_ms ASC LIMIT ?";

    sqlite3_stmt* stmt = nullptr; // SQLite 预处理语句句柄
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false; // SQL 编译失败
    }

    // 绑定 limit 参数（索引从 1 开始，? 的位置为第 1 个参数）
    sqlite3_bind_int(stmt, 1, limit);
    // 逐行读取查询结果
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        UploadRecord rec;
        rec.id = sqlite3_column_int64(stmt, 0);            // 主键 ID（第 0 列）
        rec.ts_ms = sqlite3_column_int64(stmt, 1);         // 时间戳（毫秒）（第 1 列）

        const unsigned char* type_text = sqlite3_column_text(stmt, 2);
        if (type_text) {
            rec.type = reinterpret_cast<const char*>(type_text); // 记录类型（第 2 列）
        }

        const unsigned char* payload_text = sqlite3_column_text(stmt, 3);
        if (payload_text) {
            rec.payload = reinterpret_cast<const char*>(payload_text); // JSON 负载（第 3 列）
        }

        rec.retry_count = sqlite3_column_int(stmt, 4);     // 重试次数（第 4 列）
        rec.last_try_ts = sqlite3_column_int64(stmt, 5);   // 最后尝试时间（第 5 列）
        out.push_back(std::move(rec)); // 移动语义追加到输出列表
    }

    sqlite3_finalize(stmt); // 释放预处理语句句柄
    return !out.empty();    // 有记录返回 true，空列表返回 false
#endif
}

// 将指定 ID 的记录批量标记为"已上传"（uploaded=1）
// 使用分批 UPDATE + 事务，避免单次 SQL 过长或锁表时间过久
// 参数 ids: 要标记为已上传的记录 ID 列表
// 返回值: true=全部标记成功，false=失败（事务已回滚）
bool SqliteStore::markUploaded(const std::vector<std::int64_t>& ids) {
    if (!config_.enable) {
        return false; // 存储未启用
    }

#ifndef HAVE_SQLITE3
    return false;
#else
    std::lock_guard<std::mutex> lock(db_mutex_); // 获取数据库锁
    flushPendingLocked(config_.flush_before_mark); // 可配置：标记前是否先刷盘
    if (!db_ || ids.empty()) {
        return false; // 数据库未打开或无 ID 列表
    }

    // 按配置的分块大小分批执行 UPDATE，避免单条 SQL 太长
    const int chunk_size = std::max(1, config_.mark_update_chunk_size);
    // 开启即时事务（IMMEDIATE 模式获取写锁，防止死锁）
    if (sqlite3_exec(db_, "BEGIN IMMEDIATE TRANSACTION;", nullptr, nullptr, nullptr) != SQLITE_OK) {
        return false; // 事务开启失败
    }

    bool ok = true;
    // 按 chunk_size 分批处理
    for (std::size_t offset = 0; offset < ids.size(); offset += static_cast<std::size_t>(chunk_size)) {
        const std::size_t chunk_len = std::min(static_cast<std::size_t>(chunk_size), ids.size() - offset);
        // 构建 SQL：UPDATE upload_queue SET uploaded=1 WHERE id IN (?,?,? ...)
        const std::string sql =
            "UPDATE upload_queue SET uploaded=1 WHERE id IN (" + buildPlaceholders(chunk_len) + ")";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            ok = false;
            break; // SQL 编译失败，跳出循环回滚
        }

        // 绑定所有 ID 参数（索引从 1 开始）
        for (std::size_t i = 0; i < chunk_len; ++i) {
            sqlite3_bind_int64(stmt, static_cast<int>(i + 1), ids[offset + i]);
        }

        const int rc = sqlite3_step(stmt); // 执行 UPDATE
        sqlite3_finalize(stmt);           // 释放语句句柄
        if (rc != SQLITE_DONE && rc != SQLITE_ROW) {
            ok = false; // 执行失败
            break;
        }
    }

    if (!ok) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr); // 回滚事务
        return false;
    }

    // 提交事务
    if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr); // COMMIT 失败也回滚
        return false;
    }

    // 更新监控统计计数器（原子操作，线程安全）
    mark_count_.fetch_add(static_cast<std::uint64_t>(ids.size()), std::memory_order_relaxed);
    return true;
#endif
}

// 将指定 ID 的记录标记为"上传失败"：递增重试次数，记录错误信息和尝试时间
// 参数 ids: 要标记失败的记录 ID 列表
// 参数 error: 失败原因描述（如 "timeout"/"network_error" 等）
// 参数 ts_ms: 当前尝试的时间戳（毫秒）
// 返回值: true=标记成功，false=失败
bool SqliteStore::markFailed(const std::vector<std::int64_t>& ids,
                             const std::string& error,
                             std::int64_t ts_ms) {
    if (!config_.enable) {
        return false; // 存储未启用
    }

#ifndef HAVE_SQLITE3
    return false;
#else
    std::lock_guard<std::mutex> lock(db_mutex_); // 获取数据库锁
    flushPendingLocked(config_.flush_before_mark); // 可配置：标记前先刷盘
    if (!db_ || ids.empty()) {
        return false; // 数据库未打开或无 ID 列表
    }

    const int chunk_size = std::max(1, config_.mark_update_chunk_size);
    // 开启即时事务
    if (sqlite3_exec(db_, "BEGIN IMMEDIATE TRANSACTION;", nullptr, nullptr, nullptr) != SQLITE_OK) {
        return false;
    }

    bool ok = true;
    for (std::size_t offset = 0; offset < ids.size(); offset += static_cast<std::size_t>(chunk_size)) {
        const std::size_t chunk_len = std::min(static_cast<std::size_t>(chunk_size), ids.size() - offset);
        // SQL：UPDATE retry_count=retry_count+1, last_try_ts=?, last_error=? WHERE id IN (...)
        const std::string sql =
            "UPDATE upload_queue SET retry_count = retry_count + 1, last_try_ts = ?, last_error = ? WHERE id IN ("
            + buildPlaceholders(chunk_len) + ")";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            ok = false;
            break;
        }

        // 绑定参数：位置 1 = ts_ms（尝试时间）, 位置 2 = error（错误信息）
        sqlite3_bind_int64(stmt, 1, ts_ms);
        // SQLITE_TRANSIENT: 告诉 SQLite 在 step 之前拷贝字符串（因为 error.c_str() 的生命周期可能短于 stmt）
        sqlite3_bind_text(stmt, 2, error.c_str(), -1, SQLITE_TRANSIENT);
        // 后续位置绑定 ID 列表（从位置 3 开始）
        for (std::size_t i = 0; i < chunk_len; ++i) {
            sqlite3_bind_int64(stmt, static_cast<int>(3 + i), ids[offset + i]);
        }

        const int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (rc != SQLITE_DONE && rc != SQLITE_ROW) {
            ok = false;
            break;
        }
    }

    if (!ok) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr); // 回滚
        return false;
    }

    // 提交事务
    if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
    }

    mark_count_.fetch_add(static_cast<std::uint64_t>(ids.size()), std::memory_order_relaxed);
    return true;
#endif
}

// 将指定 ID 的记录标记为"死亡"（uploaded=2）：永久放弃上传
// 用于超过最大重试次数、数据格式错误等不可恢复场景
// 参数 ids: 要标记为死亡的记录 ID 列表
// 参数 error: 标记死亡的原因描述
// 返回值: true=标记成功，false=失败
bool SqliteStore::markDead(const std::vector<std::int64_t>& ids,
                           const std::string& error) {
    if (!config_.enable) {
        return false; // 存储未启用
    }

#ifndef HAVE_SQLITE3
    return false;
#else
    std::lock_guard<std::mutex> lock(db_mutex_); // 获取数据库锁
    flushPendingLocked(config_.flush_before_mark); // 可配置：标记前先刷盘
    if (!db_ || ids.empty()) {
        return false; // 数据库未打开或无 ID 列表
    }

    const int chunk_size = std::max(1, config_.mark_update_chunk_size);
    // 开启即时事务
    if (sqlite3_exec(db_, "BEGIN IMMEDIATE TRANSACTION;", nullptr, nullptr, nullptr) != SQLITE_OK) {
        return false;
    }

    const std::int64_t now_ms = nowMs(); // 当前时间作为最后尝试时间
    bool ok = true;
    for (std::size_t offset = 0; offset < ids.size(); offset += static_cast<std::size_t>(chunk_size)) {
        const std::size_t chunk_len = std::min(static_cast<std::size_t>(chunk_size), ids.size() - offset);
        // SQL：设置 uploaded=2（死亡状态）、记录最后尝试时间和错误信息
        const std::string sql =
            "UPDATE upload_queue SET uploaded = 2, last_try_ts = ?, last_error = ? WHERE id IN ("
            + buildPlaceholders(chunk_len) + ")";

        sqlite3_stmt* stmt_dead = nullptr; // 使用 stmt_dead 命名以区分散落在各处的 stmt 变量
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt_dead, nullptr) != SQLITE_OK) {
            ok = false;
            break;
        }

        // 绑定参数：位置 1 = 当前时间戳, 位置 2 = 死亡原因
        sqlite3_bind_int64(stmt_dead, 1, now_ms);
        sqlite3_bind_text(stmt_dead, 2, error.c_str(), -1, SQLITE_TRANSIENT);
        // 后续位置绑定 ID 列表（从位置 3 开始）
        for (std::size_t i = 0; i < chunk_len; ++i) {
            sqlite3_bind_int64(stmt_dead, static_cast<int>(3 + i), ids[offset + i]);
        }

        const int rc = sqlite3_step(stmt_dead);
        sqlite3_finalize(stmt_dead);
        if (rc != SQLITE_DONE && rc != SQLITE_ROW) {
            ok = false;
            break;
        }
    }

    if (!ok) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr); // 回滚
        return false;
    }

    // 提交事务
    if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
    }

    mark_count_.fetch_add(static_cast<std::uint64_t>(ids.size()), std::memory_order_relaxed);
    return true;
#endif
}

// 查询当前待上传记录的数量（uploaded=0）
// 返回值: 未上传记录总数，出错或未启用时返回 0
int SqliteStore::pendingUploadCount() {
    if (!config_.enable) {
        return 0; // 存储未启用
    }

#ifndef HAVE_SQLITE3
    return 0; // SQLite3 不可用
#else
    std::lock_guard<std::mutex> lock(db_mutex_); // 获取数据库锁
    flushPendingLocked(false); // 不强制刷盘，仅查询已持久化的数据
    if (!db_) {
        return 0; // 数据库未打开
    }

    // SQL：统计 uploaded=0 的记录数（COUNT 聚合查询，带索引 idx_upload_queue_uploaded 加速）
    const char* sql = "SELECT COUNT(1) FROM upload_queue WHERE uploaded=0";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return 0; // SQL 编译失败
    }

    int pending = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        pending = sqlite3_column_int(stmt, 0); // 读取 COUNT 结果
    }
    sqlite3_finalize(stmt); // 释放语句句柄
    return pending;
#endif
}

// 获取运行时统计信息（线程安全读取原子变量）
SqliteStore::RuntimeStats SqliteStore::runtimeStats() const {
    RuntimeStats stats;
    stats.last_flush_us = last_flush_us_.load(std::memory_order_relaxed);   // 最近刷盘耗时
    stats.flush_count = flush_count_.load(std::memory_order_relaxed);       // 累计刷盘次数
    stats.mark_count = mark_count_.load(std::memory_order_relaxed);         // 累计标记条数
    return stats;
}

// 根据时间戳打开对应的数据库文件（按天分库策略）
// 如果日期与当前已打开的库不同，自动关闭旧库、打开新库
// 参数 ts_ms: 时间戳（毫秒），用于确定目标数据库日期
// 返回值: true=数据库已就绪，false=打开失败
bool SqliteStore::openForTimestamp(std::int64_t ts_ms) {
#ifndef HAVE_SQLITE3
    return false; // SQLite3 不可用
#else
    // 计算目标日期字符串（YYYYMMDD）
    const std::string day = toDayString(ts_ms);
    // 如果日期相同且数据库已打开，无需操作
    if (day == current_day_ && db_) {
        return true;
    }

    // 关闭旧连接，准备切换到新日期的数据库
    if (db_) {
        // 关闭前执行 WAL 检查点（TRUNCATE 模式：将 WAL 数据写入主库并截断 WAL 文件）
        // 防止 WAL 文件无限增长
        sqlite3_exec(db_, "PRAGMA wal_checkpoint(TRUNCATE);", nullptr, nullptr, nullptr);
        sqlite3_close(db_); // 关闭旧数据库连接
        db_ = nullptr;
    }

    // 构建新数据库文件的完整路径并打开
    const std::string path = buildDbPath(day);
    sqlite3* handle = nullptr;
    if (sqlite3_open(path.c_str(), &handle) != SQLITE_OK) { // SQLite3 打开/创建数据库文件
        std::cerr << "SqliteStore: open failed: " << path << std::endl;
        return false;
    }

    db_ = handle;          // 保存新连接句柄
    current_day_ = day;    // 更新当前日期

    // 配置 WAL 模式（Write-Ahead Logging，提高并发写入性能，读写不互斥）
    if (config_.wal) {
        sqlite3_exec(handle, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
    }
    // 配置同步级别（OFF: 最快但断电可能损坏 / NORMAL: 平衡 / FULL: 最安全）
    if (!config_.sync.empty()) {
        const std::string pragma = "PRAGMA synchronous=" + config_.sync + ";";
        sqlite3_exec(handle, pragma.c_str(), nullptr, nullptr, nullptr);
    }

    // 确保表结构存在（CREATE TABLE IF NOT EXISTS）
    return createTables();
#endif
}

// 创建业务表结构（upload_queue 表及其索引）
// 同时兼容历史数据库：尝试 ALTER TABLE 补充缺失列
// 返回值: true=表已就绪，false=创建失败
bool SqliteStore::createTables() {
#ifndef HAVE_SQLITE3
    return false;
#else
    if (!db_) {
        return false; // 数据库未打开
    }

    // 创建上传队列表：用于缓存所有待上传的业务记录
    // 字段说明：
    //   id:           自增主键
    //   ts_ms:        事件时间戳（毫秒），用于服务端时间排序
    //   type:         记录类型（inference/sensor/device_status/log/frame_meta）
    //   payload:      JSON 格式的数据负载
    //   uploaded:     上传状态 0=未上传, 1=已上传, 2=已死亡
    //   retry_count:  重试次数（每次上传失败后递增）
    //   last_try_ts:  最近一次重试时间
    //   last_error:   最近一次失败原因
    const char* sql =
        "CREATE TABLE IF NOT EXISTS upload_queue ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"  // 自增主键
        "ts_ms INTEGER NOT NULL,"                // 事件时间戳，非空
        "type TEXT NOT NULL,"                    // 记录类型，非空
        "payload TEXT NOT NULL,"                 // JSON 负载，非空
        "uploaded INTEGER DEFAULT 0,"            // 上传状态，默认 0（未上传）
        "retry_count INTEGER DEFAULT 0,"         // 重试次数，默认 0
        "last_try_ts INTEGER DEFAULT 0,"         // 最后尝试时间，默认 0
        "last_error TEXT DEFAULT ''"             // 错误信息，默认空
        ");"
        // 创建索引加速按时间排序的查询（fetchPending 中 ORDER BY ts_ms ASC）
        "CREATE INDEX IF NOT EXISTS idx_upload_queue_ts ON upload_queue(ts_ms);"
        // 创建索引加速按上传状态的查询（pendingUploadCount、fetchPending 中 WHERE uploaded=0）
        "CREATE INDEX IF NOT EXISTS idx_upload_queue_uploaded ON upload_queue(uploaded);";

    char* err = nullptr;
    const int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &err);
    if (err) {
        sqlite3_free(err); // 释放 SQLite 分配的错误信息
    }
    if (rc != SQLITE_OK) {
        return false; // SQL 执行失败
    }

    // 兼容历史数据库：尝试添加可能在旧版本中不存在的列
    // ALTER TABLE ADD COLUMN 在列已存在时会报错，但这里使用 sqlite3_exec 忽略错误
    // 这样新旧数据库都能正常工作
    sqlite3_exec(db_, "ALTER TABLE upload_queue ADD COLUMN retry_count INTEGER DEFAULT 0;", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "ALTER TABLE upload_queue ADD COLUMN last_try_ts INTEGER DEFAULT 0;", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "ALTER TABLE upload_queue ADD COLUMN last_error TEXT DEFAULT '';", nullptr, nullptr, nullptr);

    return true;
#endif
}

// 清理超过保留天数的历史数据库文件
// 遍历 base_dir 目录，按文件名中的日期前缀判断是否需要删除
void SqliteStore::cleanupOldFiles() {
    // retention_days <= 0 表示不清理（无限期保留）
    if (config_.retention_days <= 0) {
        return;
    }

    try {
        // 计算保留截止时间点 = 当前时间 - 保留天数 * 24 小时
        const auto now = std::chrono::system_clock::now();
        const auto cutoff = now - std::chrono::hours(24 * config_.retention_days);
        const std::time_t cutoff_t = std::chrono::system_clock::to_time_t(cutoff);

        std::tm tm_cutoff{};
        localtime_r(&cutoff_t, &tm_cutoff); // 线程安全地转换为本地时间

        // 格式化截止日期字符串（YYYYMMDD）
        char buf[16];
        std::strftime(buf, sizeof(buf), "%Y%m%d", &tm_cutoff);
        const std::string cutoff_day = buf;

        // 遍历数据库目录，删除早于截止日期的数据库文件
        for (const auto& entry : std::filesystem::directory_iterator(config_.base_dir)) {
            if (!entry.is_regular_file()) {
                continue; // 跳过非普通文件（目录、符号链接等）
            }

            const std::string name = entry.path().filename().string();
            // 仅处理以 db_prefix 开头的文件（匹配我们的数据库命名规则）
            if (name.rfind(config_.db_prefix, 0) != 0) {
                continue; // 文件名前缀不匹配，跳过
            }

            // 提取文件名中的 8 位日期部分
            const std::string day = name.substr(config_.db_prefix.size(), 8);
            // 日期字符串长度为 8 且小于截止日期时删除
            if (day.size() == 8 && day < cutoff_day) {
                std::filesystem::remove(entry.path()); // 删除过期文件
            }
        }
    } catch (...) {
        // 捕获所有异常（如目录不存在、权限不足等），避免影响主流程
        std::cerr << "SqliteStore: cleanup failed" << std::endl;
    }
}

// 根据日期字符串构造数据库文件的完整路径
// 参数 day: 日期字符串（格式 YYYYMMDD）
// 返回值: 完整的数据库文件路径，如 "../data/sqlite/edge_data_20260105.db"
std::string SqliteStore::buildDbPath(const std::string& day) const {
    std::ostringstream oss;
    oss << config_.base_dir;
    // 如果 base_dir 不以 '/' 或 '\' 结尾，追加路径分隔符
    if (!config_.base_dir.empty() && config_.base_dir.back() != '/' && config_.base_dir.back() != '\\') {
        oss << "/";
    }
    oss << config_.db_prefix << day << ".db"; // 拼接：前缀 + 日期 + .db 扩展名
    return oss.str();
}

// 将毫秒时间戳转换为日期字符串（YYYYMMDD 格式）
// 参数 ts_ms: 毫秒级 Unix 时间戳
// 返回值: 8 位日期字符串
std::string SqliteStore::toDayString(std::int64_t ts_ms) const {
    const std::time_t t = static_cast<std::time_t>(ts_ms / 1000); // 毫秒转秒
    std::tm tm_time{};
    localtime_r(&t, &tm_time); // 线程安全地转换为本地时间结构体

    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y%m%d", &tm_time); // 格式化为 YYYYMMDD
    return std::string(buf);
}

// 获取当前系统时间的毫秒值（封装 toUnixMs 调用）
// 返回值: 当前时间的毫秒级 Unix 时间戳
std::int64_t SqliteStore::nowMs() const {
    return toUnixMs(std::chrono::system_clock::now());
}

// 插入单条记录的内部实现（快速路径 + 慢路径的分层设计）
// 快速路径：仅持有写缓冲锁，将数据推入 pending_writes_
// 慢路径：当缓冲区达到批量阈值或时间阈值时，获取数据库锁并执行批量写入
// 参数 type: 记录类型
// 参数 ts_ms: 事件时间戳（毫秒）
// 参数 payload: JSON 格式负载
// 返回值: true=成功写入缓冲或数据库，false=失败
bool SqliteStore::insertRecord(const std::string& type, std::int64_t ts_ms, const std::string& payload) {
#ifndef HAVE_SQLITE3
    return false; // SQLite3 不可用
#else
    // ==================== 阶段 1: 快速路径（仅持有写缓冲锁） ====================
    bool need_flush = false;     // 是否需要触发刷盘
    bool need_day_switch = false; // 是否需要切换数据库（跨天）
    {
        std::lock_guard<std::mutex> wlock(write_buf_mutex_); // 仅获取写缓冲锁，不阻塞数据库查询

        // 检查是否跨天：如果已存在缓冲数据且新记录日期与当前数据库日期不同，
        // 需要先刷盘再切换数据库
        const std::string day = toDayString(ts_ms);
        need_day_switch = !current_day_.empty() && day != current_day_ && !pending_writes_.empty();

        // 将记录推入写缓冲队列
        PendingWrite rec;
        rec.type = type;
        rec.ts_ms = ts_ms;
        rec.payload = payload;
        pending_writes_.push_back(std::move(rec));

        // 检查是否达到刷盘触发条件（三选一）：
        //   1. 缓冲区大小 >= write_batch_size（批量阈值）
        //   2. 距上次刷盘时间 >= write_flush_ms（时间阈值）
        //   3. 需要跨天切库（跨天场景强制刷盘）
        const int batch_size = (config_.write_batch_size > 0) ? config_.write_batch_size : 1;
        const std::int64_t now_ms = nowMs();
        const bool size_reached = static_cast<int>(pending_writes_.size()) >= batch_size; // 数量条件
        const bool time_reached =
            (config_.write_flush_ms <= 0) ||        // 时间阈值未设置（<=0 视为总是满足）
            (last_flush_ms_ <= 0) ||                // 尚未记录首次刷新时间
            ((now_ms - last_flush_ms_) >= config_.write_flush_ms); // 距上次刷新已超时
        need_flush = need_day_switch || size_reached || time_reached; // 任一条件满足则需刷盘
    }

    // ==================== 阶段 2: 慢路径（需要时获取数据库锁） ====================
    if (!need_flush) {
        return true; // 缓冲区未满且未超时，数据已安全暂存于内存中
    }

    // 获取数据库锁执行批量写入
    std::lock_guard<std::mutex> dlock(db_mutex_);
    // 确保目标日期的数据库已打开（可能在快速路径中检测到跨天）
    if (!openForTimestamp(ts_ms)) {
        return false;
    }
    // 如果刚发生了跨天切库，顺便清理旧文件
    if (need_day_switch) {
        cleanupOldFiles();
    }
    // 执行批量刷盘操作（force=true 强制写入）
    return flushPendingLocked(true);
#endif
}

// 刷新批量写缓冲区到 SQLite 数据库（核心写入逻辑）
// 注意：调用方必须已持有 db_mutex_（数据库互斥锁）
// 参数 force: true=强制刷盘（忽略阈值条件），false=仅在达到阈值时才刷盘
// 返回值: true=刷盘成功（或缓冲区为空），false=写入失败
bool SqliteStore::flushPendingLocked(bool force) {
#ifndef HAVE_SQLITE3
    return false;
#else
    // 注意：调用方必须已持有 db_mutex_
    // 这里获取 write_buf_mutex_ 来排空缓冲区（双锁设计：先 db_mutex_ 再 write_buf_mutex_）
    std::vector<PendingWrite> batch;
    {
        std::lock_guard<std::mutex> wlock(write_buf_mutex_);

        if (pending_writes_.empty()) {
            return true; // 缓冲区为空，无需操作
        }

        // 再次计算是否达到批量写入条件（防止在获取 db_mutex_ 期间条件已变化）
        const int batch_size = (config_.write_batch_size > 0) ? config_.write_batch_size : 1;
        const std::int64_t now_ms = nowMs();
        const bool size_reached = static_cast<int>(pending_writes_.size()) >= batch_size; // 数量条件
        const bool time_reached =
            (config_.write_flush_ms <= 0) ||
            (last_flush_ms_ <= 0) ||
            ((now_ms - last_flush_ms_) >= config_.write_flush_ms); // 时间条件

        // 非强制刷盘且未达到触发条件时直接返回（已经在锁内再次确认）
        if (!force && !size_reached && !time_reached) {
            return true;
        }

        // 原子交换：将缓冲区全部移出到本地 batch，清空 pending_writes_
        // 这样在后续写入过程中，新的 insertRecord 调用可以继续向新的缓冲区推入数据
        batch.swap(pending_writes_);
        last_flush_ms_ = nowMs(); // 更新上次刷盘时间
    }

    const auto flush_begin = std::chrono::steady_clock::now(); // 记录刷盘开始时间（用于耗时统计）

    // 确保数据库已打开（如果因为某种原因关闭了，用 batch 中首条记录的时间戳重试打开）
    if (!db_) {
        if (!openForTimestamp(batch.front().ts_ms)) {
            return false;
        }
    }

    // 开启即时事务（IMMEDIATE：立即获取写锁，防止死锁）
    // 使用事务批量写入可以大幅提升性能（减少 fsync 次数）
    if (sqlite3_exec(db_, "BEGIN IMMEDIATE TRANSACTION;", nullptr, nullptr, nullptr) != SQLITE_OK) {
        return false;
    }

    // 准备 INSERT SQL 语句
    // 新记录状态默认值：uploaded=0(未上传), retry_count=0, last_try_ts=0, last_error=''
    const char* sql =
        "INSERT INTO upload_queue (ts_ms, type, payload, uploaded, retry_count, last_try_ts, last_error) "
        "VALUES (?, ?, ?, 0, 0, 0, '')";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr); // SQL 编译失败，回滚事务
        return false;
    }

    // 逐条绑定参数并执行插入
    bool ok = true;
    for (const PendingWrite& rec : batch) {
        sqlite3_reset(stmt);           // 重置语句状态（必须在 sqlite3_clear_bindings 之前）
        sqlite3_clear_bindings(stmt);  // 清除之前的参数绑定

        // 绑定 3 个参数值
        sqlite3_bind_int64(stmt, 1, rec.ts_ms);   // 位置 1: 时间戳（毫秒）
        // SQLITE_STATIC: batch 的生命周期覆盖整个循环和 step 调用，SQLite 不需要额外拷贝字符串
        sqlite3_bind_text(stmt, 2, rec.type.c_str(), -1, SQLITE_STATIC);    // 位置 2: 记录类型
        sqlite3_bind_text(stmt, 3, rec.payload.c_str(), -1, SQLITE_STATIC); // 位置 3: JSON 负载

        const int rc = sqlite3_step(stmt); // 执行 INSERT
        if (rc != SQLITE_DONE) {           // 期望返回 SQLITE_DONE（插入完成）
            ok = false;
            break; // 插入失败，跳出循环回滚
        }
    }

    sqlite3_finalize(stmt); // 释放预处理语句

    if (ok) {
        // 提交事务：将缓冲区的所有数据一次性持久化到数据库
        if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK) {
            sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr); // COMMIT 失败时回滚
            return false;
        }
        // 统计刷盘耗时（微秒）和次数（用于运行时监控）
        const auto flush_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - flush_begin).count();
        last_flush_us_.store(flush_us, std::memory_order_relaxed);  // 更新最近刷盘耗时
        flush_count_.fetch_add(1, std::memory_order_relaxed);       // 递增刷盘次数
        return true;
    }

    // 插入失败，回滚事务（数据会丢失，但 pending_writes_ 已在 swap 时排空）
    sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
    return false;
#endif
}

// 基础 JSON 字符串转义工具
// 对 JSON 中有特殊含义的字符进行转义，防止破坏 JSON 结构
// 参数 input: 原始字符串
// 返回值: 转义后的安全字符串
std::string SqliteStore::escapeJson(const std::string& input) const {
    std::string out;
    out.reserve(input.size()); // 预分配空间（大多数情况输入不需要转义）
    for (char ch : input) {
        if (ch == '"') {
            out += "\\\"";   // 双引号转义：\"
        } else if (ch == '\\') {
            out += "\\\\";   // 反斜杠转义：\\
        } else if (ch == '\n') {
            out += "\\n";    // 换行符转义：\n
        } else if (ch == '\r') {
            out += "\\r";    // 回车符转义：\r
        } else if (ch == '\t') {
            out += "\\t";    // 制表符转义：\t
        } else {
            out += ch;       // 普通字符直接追加
        }
    }
    return out;
}

} // namespace data_lifecycle
