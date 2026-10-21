#pragma once

#include "core/basic/core.h"
#include "core/geometry/ray.h"
#include "core/scene/scene.h"
#include "core/tracer/camera.h"
#include "core/medium/medium.h"

namespace mps::core {

  struct TraceSettings {
    i32 maxBounce = 10;
    f32 eps      = 1e-3f;   // offset off a surface
    f32 nearEps  = 1e-2f;   // ray.min after a bounce
  };

  // Immutable POD snapshot of everything one path needs.
  struct RenderContext {
    SceneDevice scene;
    Medium medium;
    bool hasMedium = false;
    TraceSettings settings;
  };

  // Background radiance for rays that escape the scene (simple gradient sky).
  MPHD MPINL RGB_F32 background(const vec3f& d) {
    f32 t = 0.5f * (d.y + 1.0f);
    return { mps::lerpf(0.05f, 0.40f, t),
             mps::lerpf(0.08f, 0.50f, t),
             mps::lerpf(0.15f, 0.70f, t) };
  }

  MPHD RGB_F32 trace(const Ray& ray0, const RenderContext& ctx, u32 seed);

}
