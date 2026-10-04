#pragma once

#include <cstdint>
#include "core/basic/attr.h"

namespace mps {


  using u32 = uint32_t;
  using u16 = uint16_t; 
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

  struct RGB_Radiance {
    f32 r,g,b; 
  }; 

  template<typename PixelType> 
  struct FrameBuffer {
    i32 width, height; 
    PixelType* pixels; 

    MPHD MPINL PixelType& pixel(i32 x, i32 y) {return pixels[y * width + x]; }
    MPHD MPINL u32 pixelnum() const {return width*height; }
    MPHD MPINL u32 bytesize() const {return width*height*sizeof(PixelType); }  
  }; 


}
