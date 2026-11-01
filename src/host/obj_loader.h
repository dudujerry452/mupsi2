#pragma once

#include <string>
#include <vector>

#include "core/basic/types.h"

namespace mps::host {

  // Minimal Wavefront OBJ loader: vertices + triangle faces only (with optional
  // v/vt/vn and v//vn forms). Good enough for the test geometry in ~/models.
  struct ObjMesh {
    std::vector<vec3f> vertices;
    std::vector<u32>   faces;   // 3 × u32 per triangle (0-based indices)
  };

  // Returns false if the file cannot be opened or contains no triangles.
  bool loadObj(const std::string& path, ObjMesh& out);

  // Auto-fit convention shared by every scene: centre the mesh, scale its
  // bounding-box diagonal to 2*targetRadius, then drop it so its lowest vertex
  // sits exactly on `floorY`. Returns the transformed vertices.
  //
  // Both scene_builder (one object) and mesh2gp's comparison scene (two objects)
  // use this so the two placements can never drift apart. Earlier each site had
  // its own copy and they disagreed about the floor plane, which left the two
  // objects standing at different heights.
  std::vector<vec3f> fitMeshToRadius(const std::vector<vec3f>& src, f32 targetRadius,
                                     const vec3f& target, f32 floorY);

}
