#pragma once 

#include <string>
#include "core/basic/types.h"

#include "core/tracer/texture.h"

namespace mps::host {
  MPH MPINL RGB_8Bit radianceTo8bit(RGB_8Bit rgb); 
  MPH MPINL RGB_F32 bit8ToRadiance(RGB_8Bit rgb); 

  MPH void radianceTo8bit(Texture<RGB_8Bit>& out, const Texture<RGB_F32>& fb); 

  MPH void savePPM(const std::string& filename, const Texture<RGB_8Bit>& fb); 

}