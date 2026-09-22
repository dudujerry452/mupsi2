#include "kernels/test/dispatch.h"

#include "core/math/random.h"

namespace mps::kernels {

  void test_func<Backend::CPU>::run(i32 n, u32* out) { 
    #pragma omp parallel for 
    for(i32 i = 0; i < n; i ++) {
      out[i] = core::xxhash(i); 
    }
  }

}