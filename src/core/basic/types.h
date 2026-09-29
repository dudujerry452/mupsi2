#pragma once

#include <cstdint>
#include "core/basic/attr.h"

namespace mps {


  using u32 = uint32_t;
  using u16 = uint16_t; 
  using i32 = int32_t;
  using i16 = int16_t; 
  using f32 = float;

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


}
