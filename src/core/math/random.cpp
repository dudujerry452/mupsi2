#include "core/math/random.h"

namespace mps::core {

  constexpr u32 PRIME32_2 = 2246822519U;
  constexpr u32 PRIME32_3 = 3266489917U;
  constexpr u32 PRIME32_4 = 668265263U;
  constexpr u32 PRIME32_5 = 374761393U;

  MPHD void combine(u32& seed, u32 val) {
    seed ^= val  + 0x9e3779b0 + (seed << 6) + (seed >> 2);
  }

  MPHD u32 xxhash(u32 x) {
      u32 h = x + PRIME32_5;
      h = PRIME32_4 * ((h << 17) | (h >> 15));
      h = PRIME32_2 * (h ^ (h >> 15));
      h = PRIME32_3 * (h ^ (h >> 13));
      return h ^ (h >> 16);
  }

  MPHD u32 xxhash(u32 x, u32 y) {
      u32 h = y + PRIME32_5 + x * PRIME32_3;
      h = PRIME32_4 * ((h << 17) | (h >> 15));
      h = PRIME32_2 * (h ^ (h >> 15));
      h = PRIME32_3 * (h ^ (h >> 13));
      return h ^ (h >> 16);
  }

  MPHD u32 xxhash(u32 x, u32 y, u32 z) {
      u32 h = z + PRIME32_5 + x * PRIME32_3;
      h = PRIME32_4 * ((h << 17) | (h >> 15));
      h += y * PRIME32_3;
      h = PRIME32_4 * ((h << 17) | (h >> 15));
      h = PRIME32_2 * (h ^ (h >> 15));
      h = PRIME32_3 * (h ^ (h >> 13));
      return h ^ (h >> 16);
  }

  MPHD u32 xxhash(u32 x, u32 y, u32 z, u32 w) {
      u32 h = w + PRIME32_5 + x * PRIME32_3;
      h = PRIME32_4 * ((h << 17) | (h >> 15));
      h += y * PRIME32_3;
      h = PRIME32_4 * ((h << 17) | (h >> 15));
      h += z * PRIME32_3;
      h = PRIME32_4 * ((h << 17) | (h >> 15));
      h = PRIME32_2 * (h ^ (h >> 15));
      h = PRIME32_3 * (h ^ (h >> 13));
      return h ^ (h >> 16);
  }

  #if defined(__CUDA_ARCH__)
  MPHD float u32_to_f(u32 x) {
    return cuda::std::bit_cast<float>(x);
  }
  #else
  MPHD float u32_to_f(u32 x) {
    return std::bit_cast<float>(x);
  }
  #endif

}
