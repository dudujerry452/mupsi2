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

  struct AABB {
    vec3f min; 
    vec3f max;

    MPHD vec3f length() const {
      return {max.x - min.x, max.y - min.y, max.z - min.z}; 
    }

    MPHD void update(vec3f p) {
      if(p.x > max.x) max.x = p.x; 
      else if(p.x < min.x) min.x = p.x;
      if(p.y > max.y) max.y = p.y; 
      else if(p.y < min.y) min.y = p.y;
      if(p.z > max.z) max.z = p.z; 
      else if(p.z < min.z) min.z = p.z; 
    }
  }; 

  struct BVHNode {
    AABB aabb; 

    u32 left; 
    u32 right; 

    u32 first; 
    u32 count; 
  }; 

}
