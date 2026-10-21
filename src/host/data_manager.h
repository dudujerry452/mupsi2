#pragma once

#include "core/scene/scene.h"

namespace mps::host {

  // Flatten the host staging SceneHost into owning device POD arrays.
  // Caller must call freeDevice() when done.
  core::SceneDevice toDevice(const core::SceneHost& host);

  void freeDevice(core::SceneDevice& dev);

}
