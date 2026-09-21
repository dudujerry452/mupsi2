#pragma once
#include "core/core.inl"
#include "core/types.inl"

namespace mps {

// 内核纯函数:元素级 saxpy。写一遍,CPU/GPU 都能编(MPHD = __host__ __device__)。
// 这是"算法",放 kernels/ 而不是 core/。
MPHD inline void saxpy(f32 a, f32 x, f32& y) {
    y += a * x;
}

}  // namespace mps
