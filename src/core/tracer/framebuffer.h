#pragma once 

#include "core/basic/core.h"
#include "core/utils/cuda_util.h"

namespace mps {



  template<typename PixelType> 
  struct FrameBuffer {
    i32 width, height; 
    PixelType* pixels; 

    MPHD MPINL PixelType& pixel(i32 x, i32 y) const {return pixels[y * width + x]; }
    MPHD MPINL u32 pixelnum() const {return width*height; }
    MPHD MPINL u32 bytesize() const {return width*height*sizeof(PixelType); }  

    // ---- cuda -----
    MPH void upload() {
      auto dev = ::mps::upload(pixels, pixelnum()); 
      pixels = dev; 
    }
    MPH void destory() {
      ::mps::destory(pixels); 
      pixels = nullptr; 
    }
    MPH void download() {
      pixels = ::mps::download(pixels, pixelnum());
    }

    // ---- cpu ----
    MPH void free() {
      if(pixels != nullptr) delete pixels; 
    }

  }; 


}