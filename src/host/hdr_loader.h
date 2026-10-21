#pragma once

#include <vector>
#include <string>

#include "core/basic/types.h"

namespace mps::host {

  // Radiance (.hdr / RGBE) image, decoded to linear float RGB.
  struct HdrImage {
    i32 width = 0, height = 0;
    std::vector<RGB_F32> pixels;   // width*height, row-major (top row first)
  };

  // Load a Radiance .hdr file (both flat and RLE RGBE scanlines).
  // Returns an image with width/height == 0 on failure.
  HdrImage loadHdr(const std::string& path);

}
