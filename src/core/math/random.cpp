#include "core/math/random.h"

namespace mps::core {

  constexpr u32 PRIME32_2 = 2246822519U;
  constexpr u32 PRIME32_3 = 3266489917U;
  constexpr u32 PRIME32_4 = 668265263U;
  constexpr u32 PRIME32_5 = 374761393U;

  MPHD void combine(u32& seed, u32 val) {
    // 0x9e3779b9 —— boost::hash_combine 常量,与原始 mupsi 一致
    seed ^= val + 0x9e3779b9u + (seed << 6) + (seed >> 2);
  }

  MPHD u32 make_seed_f(u32 x) {
      u32 h = x + PRIME32_5;
      h = PRIME32_4 * ((h << 17) | (h >> 15));
      h = PRIME32_2 * (h ^ (h >> 15));
      h = PRIME32_3 * (h ^ (h >> 13));
      return h ^ (h >> 16);
  }

  MPHD u32 make_seed_f(u32 x, u32 y) {
      u32 h = y + PRIME32_5 + x * PRIME32_3;
      h = PRIME32_4 * ((h << 17) | (h >> 15));
      h = PRIME32_2 * (h ^ (h >> 15));
      h = PRIME32_3 * (h ^ (h >> 13));
      return h ^ (h >> 16);
  }

  MPHD u32 make_seed_f(u32 x, u32 y, u32 z) {
      u32 h = z + PRIME32_5 + x * PRIME32_3;
      h = PRIME32_4 * ((h << 17) | (h >> 15));
      h += y * PRIME32_3;
      h = PRIME32_4 * ((h << 17) | (h >> 15));
      h = PRIME32_2 * (h ^ (h >> 15));
      h = PRIME32_3 * (h ^ (h >> 13));
      return h ^ (h >> 16);
  }

  MPHD u32 make_seed_f(u32 x, u32 y, u32 z, u32 w) {
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

  MPHD float u32_to_f(u32 x) {
    // 裸 union 位重解释:不用 cuda::std::bit_cast(libcu++ 未链接会炸)
    union { u32 u; f32 f; } c;
    c.u = x;
    return c.f;
  }

}
