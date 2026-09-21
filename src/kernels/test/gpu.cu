

#include "./dispatch.h"

#include "core/core.inl"
#include "core/random.inl"

namespace mps::kernels {

  MPG void g_calc_test(i32 n, u32* out) {
    i32 idx = blockIdx.x  * blockDim.x + threadIdx.x;  
    if(idx < n)
    out[idx] = core::xxhash(idx); 
  }

  void test_func<Backend::CUDA>::run(i32 n, u32* out) {
    u32* cudamem; 
    cudaMalloc(&cudamem, n * sizeof(u32)); 
    const i32 block = 256; 
    const i32 grid = (n+block-1) / block;
    g_calc_test<<<grid, block>>>(n, cudamem);  
    cudaMemcpy(out, cudamem, n * sizeof(u32), cudaMemcpyDeviceToHost); 
  }


}