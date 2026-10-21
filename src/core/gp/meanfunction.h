#pragma once

#include "core/math/math_util.h"

namespace mps::core {

  // Deterministic sphere SDF mean: μ(p) = ‖p - center‖ - radius.
  struct SphereMean {
    vec3f center;
    f32 radius;

    MPHD f32 eval(const vec3f& p) const {
      return mps::norm(p - center) - radius;
    }
    MPHD vec3f gradient(const vec3f& p) const {
      return mps::normalize(p - center);
    }
  };

  // NOTE: MeshMeanFunction (SDF from a triangle mesh) is deferred — the
  // infrastructure stage only needs the sphere mean. Add alongside the
  // material/mesh stage.

}
