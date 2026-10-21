#pragma once

#include "core/basic/types.h"
#include "core/math/math_util.h"

namespace mps::core {

  struct Ray {
    vec3f o;
    vec3f dir;  // must be normalized
    f32 min, max;

    MPHD MPINL vec3f at(f32 t) const { return o + t * dir; }
  };

}
