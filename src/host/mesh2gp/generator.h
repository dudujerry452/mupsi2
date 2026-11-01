#pragma once

#include <string>

#include "core/basic/types.h"
#include "host/mesh2gp/gpisfile.h"

namespace mps::host {

  // Build parameters for a mesh -> GPIS artifact.
  struct GpisBuildParams {
    // Uniform scale applied to the mesh (so ~/models-sized models land near
    // unit scale). Applied to vertices before anything else.
    f32 scale = 1.0f;

    // Sparse narrow-band SDF grid.
    u32 gridRes = 256;
    f32 bandCells = 4.0f;   // band half width, in cells

    // Kernel / noise parameters. L* (length scale) is expressed in world units
    // *after* scaling; kernelRadius defaults to L if left <= 0.
    f32 sigma = 1.0f;
    f32 Lx = 1.0f, Ly = 1.0f, Lz = 1.0f;
    f32 kernelRadius = -1.0f;   // <= 0 => use the max length scale
    u32 impulseDensity = 3;
    u32 seed = 0;
  };

  // Load `objPath`, build the pseudo-normal + BVH + sparse narrow-band SDF, and
  // return a GPISData whose blob is the exact memory image to be written.
  //
  // This is deliberately the *same* code path mupsi2 uses (SceneHost::addMesh +
  // buildSparseSDFGrid), so the artifact is guaranteed consumable by the
  // renderer's SparseSDFGrid / MeshMean without translation.
  bool buildGpisFromObj(const std::string& objPath, const GpisBuildParams& p, gpifile::GPISData& out);

} // namespace mps::host
