#pragma once

#include <cstdint>
#include "core/basic/core.h"

namespace mps {
  using u32 = uint32_t;
  using i32 = int32_t;
  using f32 = float;

  struct vec3f {f32 x,y,z; };
  MPHD vec3f operator+(const vec3f& a, const vec3f& b);
  MPHD vec3f operator*(const vec3f& a, const vec3f& b);
  MPHD vec3f operator&(const vec3f& a, const vec3f& b); // cross

}
