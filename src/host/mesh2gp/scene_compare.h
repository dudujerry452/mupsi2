#pragma once

#include <string>

#include "core/scene/scene.h"
#include "core/tracer/camera.h"
#include "core/tracer/trace.h"
#include "core/medium/medium.h"

#include "host/mesh2gp/gpisfile.h"

namespace mps::host::m2g {

  // Side-by-side comparison scene:
  //   LEFT  — the source mesh as a plain triangle mesh (no medium)
  //   RIGHT — the same mesh as a GPIS medium whose mean is the sparse SDF grid
  //           stored in the .gpibin artifact.
  //
  // Both objects are the same size and are lit the same way. The medium's
  // bounding box only encloses the right-hand object, so the left mesh is never
  // inside the medium.
  struct CompareConfig {
    core::SceneHost host;
    core::Medium    medium;
    bool            hasMedium = true;
    core::Camera    camera;
    core::TraceSettings settings;
    i32 spp = 1;
  };

  struct CompareParams {
    f32 objectRadius = 1.0f;    // radius each object is drawn at
    f32 gap          = 2.6f;    // centre-to-centre distance, in radii
    i32 width = 640, height = 360;
    f32 fovDeg = 45.0f;         // same as scene_builder
    f32 camDist = 4.0f;         // MINIMUM object distance in radii. The actual
                                //   distance grows automatically if that is not
                                //   enough to fit both objects in frame.
    f32 camHeight = 0.0f;       // extra camera height above the objects' centre,
                                //   in radii. 0 = look straight at them.
    f32 frameMargin = 1.15f;    // 15% empty border around the fitted bounds
    bool floor = true;
  };

  // `objPath` supplies the LEFT object's geometry (plain mesh path).
  // `art` supplies the RIGHT object's SDF grid and kernel parameters.
  CompareConfig buildCompareScene(const std::string& objPath,
                                  const gpifile::GPISData& art,
                                  const CompareParams& p);

} // namespace mps::host::m2g
