#pragma once

#if defined(__CUDACC__) // for .cu / nvcc

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

inline void cudaCheck(cudaError_t err) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s (code %d)\n",
                 cudaGetErrorString(err), static_cast<int>(err));
    std::abort();
  }
}

inline void cudaCheckLast() {
  cudaCheck(cudaGetLastError());
}

#endif // __CUDACC__
