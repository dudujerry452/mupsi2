#pragma once

#include <string>
#include "core/basic/types.h"
#include "core/math/math_util.h"

namespace mps::host {

  // Radiance → 8-bit sRGB (Reinhard tone map + gamma 2.2).
  MPHD MPINL RGB_8Bit radiance_to_8bit(RGB_F32 rgb) {
    auto encode = [](f32 x) -> byte {
      x = x > 0.0f ? x : 0.0f;                 // clamp
      x = x / (1.0f + x);                      // Reinhard
      x = mps::powf(x, 1.0f / 2.2f);           // gamma
      return static_cast<byte>(x * 255.0f + 0.5f);
    };
    return {encode(rgb.r), encode(rgb.g), encode(rgb.b)};
  }

  MPHD void framebuffer_radiance_to_8bit(const FrameBuffer<RGB_F32>& fb, FrameBuffer<RGB_8Bit>& ret);

  MPH void save_ppm(const std::string& filename, const FrameBuffer<RGB_8Bit>& fb);

}
