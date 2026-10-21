#pragma once

#include "core/scene/scene.h"
#include "core/medium/medium.h"

namespace mps::host {

  // Flatten the host staging SceneHost into owning device POD arrays.
  // Caller must call freeDevice() when done.
  core::SceneDevice toDevice(const core::SceneHost& host);

  void freeDevice(core::SceneDevice& dev);

  // Point a MeshMean medium at the device's shared geometry arrays, using the
  // offsets recorded in `ref` (a SceneHost::meshMeans entry). CPU uses host
  // pointers; the CUDA shell re-binds these to device pointers itself.
  void bindMeshMean(core::Medium& m, const core::SceneDevice& dev, const core::Mesh& ref);

  // Point a Voxel mean medium at the device's sparse-grid arrays (origin,
  // cellSize and band are already set on m.voxelMean by the scene builder).
  void bindVoxelMean(core::Medium& m, const core::SceneDevice& dev);

  // Point a Medium's non-stationary kernel field at the device grids.
  void bindKernelField(core::Medium& m, const core::SceneDevice& dev);

  // Point a Medium's dense RGB color grid at the device array.
  void bindColorGrid(core::Medium& m, const core::SceneDevice& dev);

}
