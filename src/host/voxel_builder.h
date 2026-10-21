#pragma once

#include <vector>

#include "core/gp/meanfunction.h"

namespace mps::host {

  // Host-side result of building a sparse narrow-band SDF grid from a mesh.
  struct VoxelGridHost {
    vec3f origin{0.0f, 0.0f, 0.0f};
    f32 cellSize = 1.0f;
    f32 band     = 1.0f;
    u32 res      = 0;

    std::vector<u8>  signGrid;   // res^3 bytes, 0 = inside (-), 1 = outside (+)
    std::vector<u32> bandIndex;  // sorted linear indices of nodes with |sdf| < band
    std::vector<f32> bandValue;  // exact signed distance, parallel to bandIndex
  };

  // Build a sparse narrow-band SDF grid by evaluating the mesh SDF at every grid
  // node (signed via the angle-weighted pseudo-normal inside MeshMean). Only the
  // narrow band |sdf| < band is kept at full precision; every node additionally
  // gets a 1-byte sign for the ±band fallback outside the band.
  //
  // `mm` must point at host-side mesh arrays (before toDevice). `bounds` is the
  // mesh's world AABB (grid = bounds + margin). OpenMP-parallel over nodes.
  VoxelGridHost buildSparseSDFGrid(const core::MeshMean& mm, const core::AABB& bounds,
                                   u32 res = 256, f32 bandCells = 4.0f);

}
