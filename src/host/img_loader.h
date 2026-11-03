#pragma once 

#include "3rdparty/stb_image.h"
#include "src/core/basic/core.h"
#include "src/core/basic/texture.h"

namespace mps::host {

  MPH Texture<RGB_F32> load_albedo(const char* path); 
  
}
