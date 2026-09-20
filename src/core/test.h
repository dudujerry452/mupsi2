#pragma once
#include "core/core.h"
#include "core/types.h"

namespace mps {

enum class Backend { CPU, CUDA };

MPHD inline void saxpy(f32 a, f32 x, f32& y) {
    y += a * x;
}

template<Backend B> struct SaxpyDispatch;

// test.cpp test.cu
template<> struct SaxpyDispatch<Backend::CPU> {
    static void run(f32 a, const f32* x, f32* y, i32 n);
};
template<> struct SaxpyDispatch<Backend::CUDA> {
    static void run(f32 a, const f32* x, f32* y, i32 n);
};

}  // namespace mps
