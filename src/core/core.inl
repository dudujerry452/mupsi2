#pragma once 
#include "core/types.inl"

#if defined(__CUDACC__) 
  #define MPH __host__
  #define MPD __device__ 
  #define MPHD __host__ __device__ 
  #define MPG __global__ 
#else 
  #define MPH 
  #define MPD 
  #define MPHD 
  #define MPG
#endif
