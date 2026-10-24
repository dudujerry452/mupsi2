#pragma once

// log.h 是 host-only 工具:std::format / std::cout / std::abort 只在 host 可用。
// device 遍(__CUDA_ARCH__ 定义)里把 host 实现藏起来,避免 clang 的 device 遍
// 对 __host__ 函数体里的 std::string_view 等 host-only 用法报假阳性。

#if !defined(__CUDA_ARCH__)

#include <iostream>
#include <format>
#include <chrono>
#include <string_view>
#include <cstdlib>   // std::abort

namespace mps {

    template <typename... Args>
    void _impl_log(std::string_view prefix, std::string_view fmt, Args&&... args) {
        auto now = std::chrono::system_clock::now();
        auto time_str = std::format("{:%Y-%m-%d %H:%M:%S}", now);

        std::string message = std::vformat(fmt, std::make_format_args(args...));
        std::cout << std::format("[{}] [{}] {}\n", time_str, prefix, message);
    }

}

#if defined(__GNUC__) || defined(__clang__)
    #define CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
    #define CURRENT_FUNCTION __FUNCSIG__
#else
    #define CURRENT_FUNCTION __func__
#endif

// 日志:[时间戳] [函数名] 消息
#define mplog(fmt, ...) \
    mps::_impl_log(CURRENT_FUNCTION, fmt, ##__VA_ARGS__)

// 日志:自定义 tag 作前缀
#define mplogtag(tag, fmt, ...) \
    mps::_impl_log(tag, fmt, ##__VA_ARGS__)

// lpanic(cond, fmt, ...):cond 为真时,记录一条 [FATAL] 日志并 abort。
// 例:
//   lpanic(ptr == nullptr, "指针为空");
//   lpanic(i > 10, "i = {} 越界", i);
#define lpanic(cond, fmt, ...) \
    do { \
        if (cond) { \
            mps::_impl_log(CURRENT_FUNCTION, "[FATAL] " fmt, ##__VA_ARGS__); \
            std::abort(); \
        } \
    } while(0)

#else  // __CUDA_ARCH__:device 遍

// 日志退化为空操作(设备端无 std::format/iostream);panic 仍保留 __trap。
#define mplog(fmt, ...) ((void)0)
#define mplogtag(tag, fmt, ...) ((void)0)
#define lpanic(cond, fmt, ...) do { if (cond) { __trap(); } } while(0)

#endif
