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
    // {value, ∇μ} in one shot.
    MPHD vec4f evalGrad(const vec3f& p) const {
      const vec3f d = p - center;
      const f32 n = mps::norm(d);
      const f32 inv = (n > 0.0f) ? 1.0f / n : 0.0f;
      return {n - radius, d.x * inv, d.y * inv, d.z * inv};
    }
  };

  // Signed-distance mesh mean: μ(p) = ± distance(p, mesh), negative inside,
  // positive outside. The sign uses the angle-weighted pseudo-normal
  // (Bærentzen & Aanæs 2005) at the closest point — robust to open meshes AND
  // locally-flipped faces (unlike a parity ray-cast, which needs watertightness,
  // or a plain face normal, which breaks on flipped faces).
  //
  // Holds raw pointers into SceneDevice's shared geometry arrays so it is
  // POD / device-friendly. On CPU these are host pointers; the CUDA shell
  // re-binds them to device pointers after upload.
  struct MeshMean {
    const vec3f*    vertices       = nullptr;  // SceneDevice::vertices
    const u32*      faces          = nullptr;  // SceneDevice::faces (3×u32)
    const BVHNode*  bvhNodes       = nullptr;  // SceneDevice::bvhNodes
    const u32*      bvhTriIndices  = nullptr;  // SceneDevice::bvhTriIndices
    const vec3f*    vertexNormals  = nullptr;  // SceneDevice::vertexNormals (pseudo-normals)
    u32 bvhRoot        = 0;                   // root node of this mesh's own BVH
    u32 bvhNodeOffset  = 0;
    u32 bvhTriOffset   = 0;

    MPHD bool empty() const {
      return vertices == nullptr || faces == nullptr || bvhNodes == nullptr ||
             bvhTriIndices == nullptr || vertexNormals == nullptr;
    }

    // {value, ∇μ} in one closest-point query.
    MPHD vec4f evalGrad(const vec3f& p) const {
      u32 tri;
      vec3f cp;
      const f32 d = closestPointBVH(p, bvhNodes + bvhNodeOffset, bvhTriIndices + bvhTriOffset,
                                    vertices, faces, bvhRoot, tri, cp);

      // Barycentric coords of cp in the closest triangle.
      const u32 i0 = faces[3 * tri + 0];
      const u32 i1 = faces[3 * tri + 1];
      const u32 i2 = faces[3 * tri + 2];
      const vec3f v0 = vertices[i0], v1 = vertices[i1], v2 = vertices[i2];
      const vec3f e0 = v1 - v0, e1 = v2 - v0, e2 = cp - v0;
      const f32 d00 = mps::dot(e0, e0), d01 = mps::dot(e0, e1), d11 = mps::dot(e1, e1);
      const f32 d20 = mps::dot(e2, e0), d21 = mps::dot(e2, e1);
      const f32 denom = d00 * d11 - d01 * d01;
      f32 u, v, w;
      if (mps::fabsf(denom) > 1e-12f) {
        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = 1.0f - v - w;
      } else { u = 1.0f; v = 0.0f; w = 0.0f; }
      u = mps::maxf(u, 0.0f); v = mps::maxf(v, 0.0f); w = mps::maxf(w, 0.0f);
      const f32 ws = u + v + w;
      if (ws > 1e-12f) { u /= ws; v /= ws; w /= ws; }

      // Interpolate angle-weighted pseudo-normal at cp.
      vec3f n = vertexNormals[i0] * u + vertexNormals[i1] * v + vertexNormals[i2] * w;
      const f32 nl = mps::norm(n);
      n = (nl > 1e-12f) ? n * (1.0f / nl) : vec3f{0.0f, 0.0f, 1.0f};

      const f32 s = (mps::dot(p - cp, n) >= 0.0f) ? 1.0f : -1.0f;
      vec3f g;
      if (d < 1e-6f) g = n;
      else g = (p - cp) * (s / d);
      return {s * d, g.x, g.y, g.z};
    }

    MPHD f32 eval(const vec3f& p) const { return evalGrad(p).x; }

    MPHD vec3f gradient(const vec3f& p) const {
      const vec4f f = evalGrad(p);
      return {f.y, f.z, f.w};
    }
  };

}
