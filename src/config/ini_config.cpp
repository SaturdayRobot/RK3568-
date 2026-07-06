/**
 * @file ini_config.cpp
 * @brief INI配置文件解析器实现
 *
 * 该文件实现了INI配置文件的解析和读取功能，支持
 * 节(section)、键值对(key=value)的解析，以及不同类型
 * 配置值的获取。
 *
 * 主要功能：
 * 1. INI文件加载和解析
 * 2. 支持不同类型配置值的获取（字符串、整数、浮点数、布尔值）
 * 3. 处理注释和空白字符
 * 4. 支持默认值
 */

#include "config/ini_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace { // 匿名命名空间，内部函数仅本翻译单元可见

/**
 * @brief 去除行内注释
 *
 * 解析策略：仅当 # 或 ; 出现在行首或紧跟在空白字符之后时，
 * 才视为注释引导符。URL等值内部不带前导空格的 #、; 会被保留，
 * 避免误截断合法值（如 rtsp://... 中的字符）。
 */
std::string stripInlineComment(const std::string& input) {
    for (size_t i = 0; i < input.size(); ++i) {
        const char ch = input[i];
        if ((ch == '#' || ch == ';') &&
            (i == 0 || std::isspace(static_cast<unsigned char>(input[i - 1])))) {
            return input.substr(0, i);
        }
    }
    return input;
}

} // namespace

bool IniConfig::load(const std::string& path) {
    // 清空旧数据，确保重复调用 load 不会累积配置项
    data_.clear();

    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::string current_section;  // 当前 [Section]，文件头部无节声明时默认为空串
    std::string line;
    while (std::getline(file, line)) {
        // 处理 UTF-8 BOM (EF BB BF)，兼容 Windows 编辑器生成的文件
        if (!line.empty() && static_cast<unsigned char>(line[0]) == 0xEF) {
            if (line.size() >= 3 &&
                static_cast<unsigned char>(line[1]) == 0xBB &&
                static_cast<unsigned char>(line[2]) == 0xBF) {
                line = line.substr(3);
            }
        }

        // 剥离注释 → 去除首尾空白 → 跳过空行
        line = stripInlineComment(line);
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        // 解析 [Section]
        if (line.front() == '[' && line.back() == ']') {
            current_section = trim(line.substr(1, line.size() - 2));
            continue;
        }

        // 解析 Key=Value：以第一个等号为界切分
        const size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            continue;  // 无等号的行视为格式非法，跳过
        }

        std::string key = trim(line.substr(0, eq_pos));
        std::string value = trim(line.substr(eq_pos + 1));
        if (key.empty()) {
            continue;  // 键为空则无法建立映射
        }

        // 存入双层哈希表: data_[section][key] = value
        data_[current_section][key] = value;
    }

    return true;
}

bool IniConfig::has(const std::string& section, const std::string& key) const {
    auto sec_it = data_.find(section);
    if (sec_it == data_.end()) {
        return false;
    }
    return sec_it->second.find(key) != sec_it->second.end();
}

std::string IniConfig::getString(const std::string& section,
                                 const std::string& key,
                                 const std::string& default_value) const {
    auto sec_it = data_.find(section);
    if (sec_it == data_.end()) {
        return default_value;
    }
    auto key_it = sec_it->second.find(key);
    if (key_it == sec_it->second.end()) {
        return default_value;
    }
    return key_it->second;
}

int IniConfig::getInt(const std::string& section, const std::string& key, int default_value) const {
    std::string value = getString(section, key, "");
    if (value.empty()) {
        return default_value;
    }
    try {
        // base=0 支持自动进制识别: 0x→十六进制, 0→八进制, 其他→十进制
        return std::stoi(value, nullptr, 0);
    } catch (...) {
        return default_value;
    }
}

double IniConfig::getDouble(const std::string& section, const std::string& key, double default_value) const {
    std::string value = getString(section, key, "");
    if (value.empty()) {
        return default_value;
    }
    try {
        return std::stod(value);
    } catch (...) {
        return default_value;
    }
}

bool IniConfig::getBool(const std::string& section, const std::string& key, bool default_value) const {
    // 转为小写以实现不区分大小写的布尔值比较
    std::string value = toLower(getString(section, key, ""));
    if (value.empty()) {
        return default_value;
    }
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    return default_value;
}

std::string IniConfig::trim(const std::string& input) {
    // 双指针法：从首尾向中间收缩，跳过空白字符
    size_t start = 0;
    while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) {
        ++start;
    }

    size_t end = input.size();
    while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }

    return input.substr(start, end - start);
}

std::string IniConfig::toLower(const std::string& input) {
    std::string out = input;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return out;
}
