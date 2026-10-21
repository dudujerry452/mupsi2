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

}
