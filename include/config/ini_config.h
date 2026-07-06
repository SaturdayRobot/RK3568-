/**
 * @file ini_config.h
 * @brief INI配置文件解析类定义
 *
 * 该文件定义了IniConfig类，负责解析和读取INI格式的配置文件，
 * 提供了获取不同类型配置值的方法，是传感器配置加载的基础类。
 *
 * 主要功能：
 * 1. 加载INI格式的配置文件
 * 2. 检查配置项是否存在
 * 3. 获取不同类型的配置值（字符串、整数、浮点数、布尔值）
 * 4. 支持默认值设置
 */

#pragma once

#include <string>
#include <unordered_map>

/**
 * @class IniConfig
 * @brief INI配置文件解析类
 *
 * 提供INI配置文件的解析和读取功能，支持多种数据类型的配置值获取。
 *
 * 数据结构：双层 unordered_map，外层 key 为 section 名，
 * 内层 key 为配置键名，value 为配置值字符串。O(1) 平均查找时间。
 */
class IniConfig {
public:
    /**
     * @brief 加载配置文件
     * @param path 配置文件路径
     * @return 加载是否成功
     */
    bool load(const std::string& path);

    /**
     * @brief 检查配置项是否存在
     */
    bool has(const std::string& section, const std::string& key) const;

    /**
     * @brief 获取字符串类型的配置值
     * @param default_value 未找到时返回的默认值（默认为空串）
     */
    std::string getString(const std::string& section,
                          const std::string& key,
                          const std::string& default_value = "") const;

    /**
     * @brief 获取整数类型的配置值
     *
     * 使用 std::stoi(value, nullptr, 0) 解析，base=0 自动识别进制：
     * 0x 前缀→十六进制，0 前缀→八进制，其他→十进制。
     * 解析失败时返回 default_value。
     */
    int getInt(const std::string& section,
               const std::string& key,
               int default_value = 0) const;

    /**
     * @brief 获取浮点数类型的配置值
     *
     * 支持科学计数法（如 1.5e3）。解析失败时返回 default_value。
     */
    double getDouble(const std::string& section,
                     const std::string& key,
                     double default_value = 0.0) const;

    /**
     * @brief 获取布尔类型的配置值
     *
     * 不区分大小写。真值关键字：1, true, yes, on
     * 假值关键字：0, false, no, off。无法识别时返回 default_value。
     */
    bool getBool(const std::string& section,
                 const std::string& key,
                 bool default_value = false) const;

private:
    // 每个 Section 是一个 key→value 的哈希映射
    using Section = std::unordered_map<std::string, std::string>;

    // 双层哈希表：section名 → (key → value)
    std::unordered_map<std::string, Section> data_;

    /** @brief 去除字符串首尾空白字符（双指针收缩法） */
    static std::string trim(const std::string& input);

    /** @brief 将字符串中所有 ASCII 字母转为小写 */
    static std::string toLower(const std::string& input);
};
