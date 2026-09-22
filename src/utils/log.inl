#pragma once 

#include <iostream>
#include <format>
#include <chrono>
#include <string_view>

template <typename... Args>
void _impl_log(std::string_view prefix, std::string_view fmt, Args&&... args) {
    auto now = std::chrono::system_clock::now();
    auto time_str = std::format("{:%Y-%m-%d %H:%M:%S}", now);

    std::string message = std::vformat(fmt, std::make_format_args(args...));
    std::cout << std::format("[{}] [{}] {}\n", time_str, prefix, message);
}

#if defined(__GNUC__) || defined(__clang__)
    #define CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
    #define CURRENT_FUNCTION __FUNCSIG__
#else
    #define CURRENT_FUNCTION __func__ // 保底方案
#endif

#define llog(fmt, ...) \
    _impl_log(CURRENT_FUNCTION, fmt, ##__VA_ARGS__)
#define llogtag(tag, fmt, ...) \
    _impl_log(tag, fmt, ##__VA_ARGS__)
