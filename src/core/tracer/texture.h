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

  // Equirectangular (lat-long) environment map for the background/IBL.
  // POD / device-friendly: pixels are host pointers on CPU; the CUDA shell
  // re-binds them after upload.
  struct EnvMap {
    i32 width = 0, height = 0;
    const RGB_F32* pixels = nullptr;   // width*height, row-major, top row first

    MPHD bool empty() const { return width == 0 || height == 0 || pixels == nullptr; }

    // Direction -> lat-long uv -> bilinear lookup (no mips, no importance
    // sampling in this stage).
    MPHD RGB_F32 sampleDir(const vec3f& d) const {
      const f32 u = 0.5f + mps::atan2f(d.z, d.x) * mps::inv_two_pi;      // [0,1)
      const f32 v = mps::acosf(mps::clampf(d.y, -1.0f, 1.0f)) * mps::inv_pi; // [0,1]

      const f32 gx = u * f32(width) - 0.5f;
      const f32 gy = v * f32(height) - 0.5f;
      i32 x0 = i32(mps::floorf(gx));
      i32 y0 = i32(mps::floorf(gy));
      const f32 tx = gx - f32(x0);
      const f32 ty = gy - f32(y0);

      const i32 x1 = x0 + 1;
      const i32 y1 = y0 + 1;

      auto texel = [&](i32 x, i32 y) -> RGB_F32 {
        x = ((x % width) + width) % width;             // wrap horizontally
        y = mps::maxf(0, mps::minf(height - 1, y));    // clamp vertically
        return pixels[size_t(y) * width + x];
      };

      const RGB_F32 a = texel(x0, y0), b = texel(x1, y0);
      const RGB_F32 c = texel(x0, y1), dd = texel(x1, y1);
      const RGB_F32 top = a * (1.0f - tx) + b * tx;
      const RGB_F32 bot = c * (1.0f - tx) + dd * tx;
      return top * (1.0f - ty) + bot * ty;
    }
  };

}
