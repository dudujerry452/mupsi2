#pragma once

#include "core/basic/types.h"

namespace mps::core {

  enum class TextureType : u16 { Constant = 0, Bitmap = 1 };

  // Minimal texture. Constant only for now (solid-color Lambertian, no material
  // assignment yet). Bitmap + loader come with the material/IO stage.
  struct Texture {
    TextureType type = TextureType::Constant;
    RGB_F32 constant = {0.8f, 0.8f, 0.8f};

    MPHD MPINL RGB_F32 sample(const vec2f& uv) const {
      (void)uv;
      return constant;
    }
  };

}
