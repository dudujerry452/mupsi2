#pragma once

#include "core/basic/core.h"
#include "core/math/math_util.h"
#include "core/math/sample.h"

namespace mps::core {

  enum class BsdfType : u16 { Null = 0, Lambertian = 1, Specular = 2 };

  struct BsdfSample {
    vec3f wi;       // scattered direction (towards next vertex)
    f32   pdf;
    RGB_F32 weight; // eval / pdf
  };

  // Solid albedo — no material/texture assignment in this milestone.
  MPHD MPINL RGB_F32 lambertianAlbedo() { return {0.72f, 0.68f, 0.62f}; }

  MPHD BsdfSample bsdfSample(BsdfType type, const vec3f& wo, const vec3f& n, const vec2f& uv,
                             f32 u1, f32 u2);
  MPHD RGB_F32   bsdfEval  (BsdfType type, const vec3f& wo, const vec3f& wi, const vec3f& n, const vec2f& uv);
  MPHD f32       bsdfPdf   (BsdfType type, const vec3f& wo, const vec3f& wi, const vec3f& n);

}
