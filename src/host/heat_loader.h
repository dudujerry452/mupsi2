#pragma once

#include <string>

#include "core/scene/scene.h"

namespace mps::host {

  // Load a precomputed heat-kernel texture (HKTEx) binary (from
  // tools/precompute_heat.py) into the SceneHost's spectral arrays.
  // Returns false on failure.
  bool loadHeatKernel(const std::string& path, core::SceneHost& host);

}
