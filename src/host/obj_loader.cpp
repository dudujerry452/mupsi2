#include "host/obj_loader.h"

#include <fstream>
#include <sstream>

namespace mps::host {

  namespace {

    // Convert a 1-based (or negative) OBJ index to a 0-based index.
    u32 normIndex(i64 idx, u64 count) {
      if (idx > 0) return u32(idx - 1);
      if (idx < 0) return u32(i64(count) + idx);
      return 0u;
    }

    // Parse one face-corner string: "v", "v/vt", "v//vn", "v/vt/vn".
    // Returns the vertex index (0 = malformed).
    i64 parseVertexIndex(const std::string& spec) {
      const size_t slash = spec.find('/');
      if (slash == std::string::npos) {
        try { return std::stoll(spec); } catch (...) { return 0; }
      }
      try { return std::stoll(spec.substr(0, slash)); } catch (...) { return 0; }
    }

  } // namespace

  bool loadObj(const std::string& path, ObjMesh& out) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    out.vertices.clear();
    out.faces.clear();

    std::string line;
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty() || line[0] == '#') continue;

      std::istringstream iss(line);
      std::string type;
      iss >> type;

      if (type == "v") {
        f32 x, y, z;
        if (iss >> x >> y >> z)
          out.vertices.push_back({x, y, z});
      } else if (type == "f") {
        std::vector<u32> vi;
        std::string spec;
        while (iss >> spec) {
          const i64 raw = parseVertexIndex(spec);
          if (raw == 0) continue;
          vi.push_back(normIndex(raw, out.vertices.size()));
        }
        if (vi.size() < 3) continue;              // skip degenerate faces
        for (size_t k = 1; k + 1 < vi.size(); ++k) {  // fan-triangulate n-gons
          out.faces.push_back(vi[0]);
          out.faces.push_back(vi[k]);
          out.faces.push_back(vi[k + 1]);
        }
      }
      // ignore vn/vt/o/g/s/usemtl/mtllib etc.
    }

    return !out.vertices.empty() && !out.faces.empty();
  }

}
