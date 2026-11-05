#pragma once

#include <cstdint>
#include "core/basic/attr.h"

namespace mps {


  using u64 = uint64_t;
  using u32 = uint32_t;
  using u16 = uint16_t; 
  using u8  = uint8_t;
  using i64 = int64_t;
  using i32 = int32_t;
  using i16 = int16_t; 
  using f32 = float;

  using byte = unsigned char; 

  struct vec2f {f32 x,y; }; 
  struct vec3f {f32 x,y,z; };
  struct vec4f {
    union {
      struct {f32 x,y,z; };
      vec3f u; 
    }; 
    f32 w; };  

  enum class Backend: u16 {
    CPU,
    CUDA
  };

  struct RGB_8Bit {
    byte r,g,b; 
  }; 

  struct RGB_F32 {
    f32 r,g,b; 
  }; 

  MPHD MPINL RGB_F32 vec3f_to_RGBF32(vec3f v) {
    RGB_F32 ret{v.x, v.y, v.z}; 
    return ret; 
  }

  MPHD MPINL vec3f RGBF32_to_vec3f(RGB_F32 rgb) {
    vec3f ret{rgb.r, rgb.g, rgb.b}; 
    return ret; 
  }

  struct TriangleU {
    u32 a; 
    u32 b; 
    u32 c; 
  }; 

}
