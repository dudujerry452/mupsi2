// 最小 CUDA smoke test:验证 nvcc 能编译 + cuBLAS 头可用。
// 正式的内核(trace/噪声/BVH)将来替换这个占位文件。
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void dummyKernel(float* out) {
    out[threadIdx.x] = threadIdx.x;
}

// 被 host 侧调用的探针:启动一次 kernel,确认设备可用。
void cudaBackendProbe(float* out, int n) {
    dummyKernel<<<1, n>>>(out);
}
