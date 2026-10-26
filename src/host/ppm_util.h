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
    mps::host::savePPM("test.ppm", fb);
    delete[] fb.pixels;
  */
  MPH MPINL RGB_8Bit radianceTo8bit(RGB_8Bit rgb); 

  MPH void radianceTo8bit(FrameBuffer<RGB_8Bit>& out, const FrameBuffer<RGB_F32>& fb); 

  MPH void savePPM(const std::string& filename, const FrameBuffer<RGB_8Bit>& fb); 

}