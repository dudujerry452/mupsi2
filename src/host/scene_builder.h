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

  // Minimal GPIS sphere scene: just the GP sphere mean + floor/wall + two area
  // lights. No extra solid spheres — ideal for isolating the effect of sigma /
  // kernel anisotropy on the GP surface shape.
  SceneConfig buildGpSphereScene(i32 w = 256, i32 h = 256, f32 sigma = 5.0f);

  // Generic: wrap any OBJ in a GP implicit surface. The mesh is auto-fit into a
  // bounding sphere of `targetRadius` centered at (0,0,-dist); camera looks
  // straight down -z from the origin. When useVoxel is true the mesh SDF is
  // precomputed into a sparse narrow-band grid (MeanType::Voxel, O(1) queries).
  SceneConfig buildMeshGpScene(const std::string& objPath, i32 w, i32 h,
                               f32 dist = 500.0f, f32 targetRadius = 150.0f,
                               core::GPMode mode = core::GPMode::SingleRealization,
                               f32 sigma = 1.0f, f32 kernelRadius = 1.0f,
                               bool useVoxel = true, u32 voxelRes = 256,
                               bool nonStationary = false, i32 sigmaFieldAxis = -1,
                               vec3f marginScale = {3.0f, 5.0f, 3.0f});

  // Convenience scenes (paths under ~/models). sigma flows into the medium bound
  // margin and the voxel narrow-band width, so it must be set at build time.
  SceneConfig buildSpotGpScene (i32 w = 256, i32 h = 256, f32 sigma = 1.0f, i32 sigmaFieldAxis = -1,
                                 vec3f marginScale = {3.0f, 5.0f, 3.0f});
  SceneConfig buildBunnyGpScene(i32 w = 512, i32 h = 512, f32 sigma = 1.0f, i32 sigmaFieldAxis = -1,
                                 vec3f marginScale = {3.0f, 5.0f, 3.0f});
  SceneConfig buildDragonGpScene(i32 w = 512, i32 h = 512, f32 sigma = 1.0f, i32 sigmaFieldAxis = -1,
                                 vec3f marginScale = {3.0f, 5.0f, 3.0f});

}
