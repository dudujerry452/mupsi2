#pragma once

#include <string>

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

  // GPIS sphere scene (wrapper sphere + GP medium + one solid Lambertian sphere).
  SceneConfig buildDefaultScene(i32 w = 160, i32 h = 120);

  // Generic: wrap any OBJ in a GP implicit surface. The mesh is auto-fit into a
  // bounding sphere of `targetRadius` centered at (0,0,-dist); camera looks
  // straight down -z from the origin.
  SceneConfig buildMeshGpScene(const std::string& objPath, i32 w, i32 h,
                               f32 dist = 500.0f, f32 targetRadius = 150.0f,
                               core::GPMode mode = core::GPMode::SingleRealization,
                               f32 sigma = 1.0f, f32 kernelRadius = 1.0f);

  // Convenience scenes (paths under ~/models).
  SceneConfig buildSpotGpScene (i32 w = 256, i32 h = 256);
  SceneConfig buildBunnyGpScene(i32 w = 512, i32 h = 512);
  SceneConfig buildDragonGpScene(i32 w = 512, i32 h = 512);

}
