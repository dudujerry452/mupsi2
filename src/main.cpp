#include <iostream>
#include <vector>
#include "kernels/saxpy_dispatch.h"

// 编译期后端选择:由 CMake 检测到 CUDA 时定义 MUPSI_USE_CUDA 决定
#if defined(MUPSI_USE_CUDA)
constexpr mps::Backend ACTIVE_BACKEND = mps::Backend::CUDA;
#else
constexpr mps::Backend ACTIVE_BACKEND = mps::Backend::CPU;
#endif

int main() {
    const mps::i32 n = 8;
    std::vector<mps::f32> x(n, 2.0f), y(n, 3.0f);
    mps::SaxpyDispatch<ACTIVE_BACKEND>::run(2.0f, x.data(), y.data(), n);
    std::cout << "y[0] = " << y[0] << " (期望 7)" << std::endl;
    return 0;
}
