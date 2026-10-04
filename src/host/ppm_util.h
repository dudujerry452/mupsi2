#pragma once 

#include <string>
#include "core/basic/types.h"

namespace mps::host {


  /*
  // standard usage:  
    mps::FrameBuffer fb; 
    fb.height = 256; 
    fb.width = 512; 
    fb.pixels = new mps::RGB[fb.pixelnum()];

    for(mps::i32 i = 0; i < fb.width; i ++) {
        for(mps::i32 j = 0; j < fb.height; j ++) {
            fb.pixel(i, j) = {static_cast<mps::byte>(i), static_cast<mps::byte>(j), 100}; 
        }
    }
    mps::host::save_ppm("test.ppm", fb);
    delete[] fb.pixels;
  */
  MPHD MPINL RGB_8Bit radiance_to_8bit(RGB_8Bit rgb); 

  MPH void framebuffer_radiance_to_8bit(const FrameBuffer<RGB_8Bit>& fb, FrameBuffer<RGB_Radiance>& ret); 

  MPH void save_ppm(const std::string& filename, const FrameBuffer<RGB_8Bit>& fb); 

}