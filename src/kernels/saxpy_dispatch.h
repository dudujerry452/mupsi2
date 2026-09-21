#pragma once
#include "kernels/saxpy.inl"

namespace mps {

enum class Backend { CPU, CUDA };

// 壳:主模板声明
template<Backend B> struct SaxpyDispatch;

// 显式特化必须在使用点之前声明,这里声明两个特化(带成员声明),
// 定义放在各自的 backend 文件里(CPU 在 .cpp, CUDA 在 .cu)。
template<> struct SaxpyDispatch<Backend::CPU> {
    static void run(f32 a, const f32* x, f32* y, i32 n);
};
template<> struct SaxpyDispatch<Backend::CUDA> {
    static void run(f32 a, const f32* x, f32* y, i32 n);
};

}  // namespace mps
