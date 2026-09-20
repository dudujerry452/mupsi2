#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "core/core.h"
#include "core/dispatch.h"

namespace mps {

__global__ void saxpyKernel(f32 a, const f32* x, f32* y, i32 n) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) saxpy(a, x[i], y[i]);
}

void SaxpyDispatch<Backend::CUDA>::run(f32 a, const f32* x, f32* y, i32 n) {
    const i32 block = 256;
    const i32 grid = (n + block - 1) / block;
    saxpyKernel<<<grid, block>>>(a, x, y, n);
}

}  // namespace mps
