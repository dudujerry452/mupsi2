#pragma once


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

#if defined(__CUDACC__)
    #define MPINL __forceinline__
#elif defined(_MSC_VER)
    #define MPINL __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define MPINL inline __attribute__((always_inline))
#else
    #define MPINL inline
#endif
