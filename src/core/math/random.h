#pragma once

#include "core/basic/types.h"

#if defined(__CUDA_ARCH__)
#include <cuda/std/bit>
#else
#include <bit>
#endif


namespace mps::core {

  MPHD void combine(u32& seed, u32 val);

  template<typename... Args>
  MPHD u32 make_seed(Args... args) {
    u32 h = 0;
    ((combine(h, static_cast<u32>(args)), ...));
    return h;
  }

  MPHD u32 xxhash(u32 x);
  MPHD u32 xxhash(u32 x, u32 y);
  MPHD u32 xxhash(u32 x, u32 y, u32 z);
  MPHD u32 xxhash(u32 x, u32 y, u32 z, u32 w);

  MPHD float u32_to_f(u32 x);

}
