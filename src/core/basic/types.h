#pragma once

#include <cstdint>
#include "core/basic/backend.h"

namespace mps {
  using u32 = uint32_t;
  using u16 = uint16_t; 
  using i32 = int32_t;
  using i16 = int16_t; 
  using f32 = float;

  struct vec3f {f32 x,y,z; };
  MPHD vec3f operator+(const vec3f& a, const vec3f& b);
  MPHD vec3f operator*(const vec3f& a, const vec3f& b);
  MPHD vec3f operator&(const vec3f& a, const vec3f& b); // cross

  struct vec4f {f32 x,y,z,w; }; 

  enum class Backend: u16 {
    CPU,
    CUDA
  };

}
