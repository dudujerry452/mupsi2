#pragma once

#include "core/basic/attr.h"
#include "core/basic/types.h"

#if defined(__CUDACC__) // only for host code in nvcc

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace mps {

  // only use for .cu

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


  template<typename T>
  MPH T* upload(T* src, u32 count) {
    T* t = nullptr; 
    u32 byte_size = count * sizeof(T); 
    cudaCheck(cudaMalloc(&t, byte_size));
    cudaCheck(cudaMemcpy(t, src, byte_size, cudaMemcpyHostToDevice)); 
    return t; 
  }
  template<typename T> 
  MPH void destory(T* item) {
    if(item != nullptr) 
      cudaCheck(cudaFree(item)); 
  }
  template<typename T> 
  MPH T* download(T* src, u32 count) {
    T* ret = new T[count]; 
    u32 byte_size = count * sizeof(T); 
    cudaCheck(cudaMemcpy(ret, src, byte_size, cudaMemcpyDeviceToHost)); 
    return ret; 
  }


}

#else

namespace mps {

  template<typename T>
  MPH T* upload(T* src, u32 count) {
    return src; 
  }
  template<typename T> 
  MPH void destory(T* item) {
    return; 
  }
  template<typename T> 
  MPH T* download(T* src, u32 count) {
    return src; 
  }

}

#endif 
