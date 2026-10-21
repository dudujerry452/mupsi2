#pragma once

#include "core/scene/scene.h"
#include "core/tracer/camera.h"
#include "core/tracer/trace.h"
#include "core/medium/medium.h"

namespace mps::host {

  // Everything needed to render one scene (host-side staging; no device upload yet).
  struct SceneConfig {
    core::SceneHost     host;
    core::Medium        medium;
    bool                hasMedium = true;
    core::Camera        camera;
    core::TraceSettings settings;
    i32                 spp = 1;
  };

  // Default GPIS sphere scene (wrapper sphere + GP medium + one solid sphere).
  SceneConfig buildDefaultScene();

  // Spot (from ~/models/spot) wrapped in a GP implicit surface — the target
  // scene for this test round. Renders at 256×256, 1 spp.
  SceneConfig buildSpotGpScene();

}
