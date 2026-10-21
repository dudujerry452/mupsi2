#pragma once

#include "core/bvh/bvh.h"
#include "core/math/math_util.h"

namespace mps::core {

  // Deterministic sphere SDF mean: μ(p) = ‖p - center‖ - radius.
  struct SphereMean {
    vec3f center;
    f32 radius;

    MPHD f32 eval(const vec3f& p) const {
      return mps::norm(p - center) - radius;
    }
    MPHD vec3f gradient(const vec3f& p) const {
      return mps::normalize(p - center);
    }
  };

  // Signed-distance mesh mean: μ(p) = ± distance(p, mesh). Negative inside,
  // positive outside (signed by a parity ray-cast, like the original mupsi).
  //
  // Holds raw pointers into SceneDevice's shared geometry arrays so it is
  // POD / device-friendly. On CPU these are host pointers; the CUDA shell
  // re-binds them to device pointers after upload.
  struct MeshMean {
    const vec3f*    vertices       = nullptr;  // SceneDevice::vertices
    const u32*      faces          = nullptr;  // SceneDevice::faces (3×u32)
    const BVHNode*  bvhNodes       = nullptr;  // SceneDevice::bvhNodes
    const u32*      bvhTriIndices  = nullptr;  // SceneDevice::bvhTriIndices
    u32 bvhRoot        = 0;                   // root node of this mesh's own BVH
    u32 bvhNodeOffset  = 0;
    u32 bvhTriOffset   = 0;

    MPHD bool empty() const {
      return vertices == nullptr || faces == nullptr || bvhNodes == nullptr || bvhTriIndices == nullptr;
    }

    MPHD bool inside(const vec3f& p) const {
      Ray r;
      r.o = p;
      r.dir = {1.0f, 0.0f, 0.0f};
      r.min = 1e-4f;
      r.max = 1e30f;
      i32 c = countIntersectionsBVH(r, bvhNodes + bvhNodeOffset, bvhTriIndices + bvhTriOffset,
                                    vertices, faces, bvhRoot);
      return (c & 1) == 1;
    }

    MPHD f32 eval(const vec3f& p) const {
      u32 tri;
      vec3f cp;
      f32 d = closestPointBVH(p, bvhNodes + bvhNodeOffset, bvhTriIndices + bvhTriOffset,
                              vertices, faces, bvhRoot, tri, cp);
      return inside(p) ? -d : d;
    }

    MPHD vec3f gradient(const vec3f& p) const {
      u32 tri;
      vec3f cp;
      f32 d = closestPointBVH(p, bvhNodes + bvhNodeOffset, bvhTriIndices + bvhTriOffset,
                              vertices, faces, bvhRoot, tri, cp);
      if (d < 1e-6f) {
        // On the surface: return the (outward, consistently wound) face normal.
        const u32 i0 = faces[3 * tri + 0];
        const u32 i1 = faces[3 * tri + 1];
        const u32 i2 = faces[3 * tri + 2];
        return mps::normalize(mps::cross(vertices[i1] - vertices[i0], vertices[i2] - vertices[i0]));
      }
      // SDF gradient points outward: (p - cp)/d points from surface to p,
      // so negate when inside.
      const f32 s = inside(p) ? -1.0f : 1.0f;
      return (p - cp) * (s / d);
    }
  };

}
