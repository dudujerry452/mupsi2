#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernels/saxpy_dispatch.h"

namespace mps {

__global__ void saxpyKernel(f32 a, const f32* x, f32* y, i32 n) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) saxpy(a, x[i], y[i]);
}

// CUDA 特化定义:负责 host↔device 数据搬运 + kernel launch
void SaxpyDispatch<Backend::CUDA>::run(f32 a, const f32* x, f32* y, i32 n) {
    f32 *dx = nullptr, *dy = nullptr;
    cudaMalloc(&dx, n * sizeof(f32));
    cudaMalloc(&dy, n * sizeof(f32));
    cudaMemcpy(dx, x, n * sizeof(f32), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, n * sizeof(f32), cudaMemcpyHostToDevice);

    const i32 block = 256;
    const i32 grid = (n + block - 1) / block;
    saxpyKernel<<<grid, block>>>(a, dx, dy, n);
    cudaDeviceSynchronize();

    cudaMemcpy(y, dy, n * sizeof(f32), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dy);
}

}  // namespace mps
