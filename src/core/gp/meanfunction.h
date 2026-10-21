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

  // Sparse narrow-band SDF grid (OpenVDB-style): the mesh SDF is precomputed
  // once. Only nodes with |sdf| < band are stored at full precision (bandIndex /
  // bandValue, sorted by linear index); every node also carries a 1-byte sign
  // (signGrid, 0 = inside / negative, 1 = outside / positive). A query at p
  // trilinearly interpolates the 8 surrounding nodes; nodes outside the band
  // fall back to ±band using their sign. This gives O(1) mean evaluation instead
  // of an O(log N) closest-point BVH traversal (paper §7, OpenVDB field).
  //
  // POD / device-friendly: pointers are host pointers on CPU; the CUDA shell
  // re-binds them to device pointers after upload.
  struct SparseSDFGrid {
    vec3f origin{0.0f, 0.0f, 0.0f};   // world position of node (0,0,0)
    f32   cellSize = 1.0f;            // world units per voxel
    f32   band     = 1.0f;            // narrow-band half width (world units)
    u32   res      = 0;               // grid resolution per axis

    const u8*  signGrid  = nullptr;   // res^3 bytes, linear idx ix+res*(iy+res*iz)
    const u32* bandIndex = nullptr;   // sorted linear indices of in-band nodes
    const f32* bandValue = nullptr;   // exact signed distance (parallel to bandIndex)
    u32 bandCount = 0;

    MPHD bool empty() const {
      return res == 0 || signGrid == nullptr || bandIndex == nullptr || bandValue == nullptr;
    }

    // Signed distance at a single grid node (linear index), with band fallback.
    MPHD f32 nodeSdf(u32 lin) const {
      // Binary search the sorted in-band node list.
      i32 lo = 0, hi = i32(bandCount) - 1;
      while (lo <= hi) {
        const i32 mid = (lo + hi) >> 1;
        const u32 key = bandIndex[mid];
        if (key == lin) return bandValue[mid];
        if (key < lin) lo = mid + 1;
        else           hi = mid - 1;
      }
      return signGrid[lin] ? band : -band;
    }

    // {value, ∇μ} in one trilinear query.
    MPHD vec4f evalGrad(const vec3f& p) const {
      const vec3f g = (p - origin) / cellSize;
      const i32 ix = i32(mps::floorf(g.x));
      const i32 iy = i32(mps::floorf(g.y));
      const i32 iz = i32(mps::floorf(g.z));
      const f32 fx = g.x - f32(ix);
      const f32 fy = g.y - f32(iy);
      const f32 fz = g.z - f32(iz);

      // Gather the 8 corner values (c[0] at (ix,iy,iz), c[k] at +bit pattern).
      f32 c[8];
      for (i32 k = 0; k < 8; ++k) {
        const i32 cx = ix + (k & 1);
        const i32 cy = iy + ((k >> 1) & 1);
        const i32 cz = iz + ((k >> 2) & 1);
        if (cx < 0 || cy < 0 || cz < 0 ||
            cx >= i32(res) || cy >= i32(res) || cz >= i32(res)) {
          c[k] = band;   // outside the grid: treat as far outside (positive)
        } else {
          const u32 lin = u32(cx) + res * (u32(cy) + res * u32(cz));
          c[k] = nodeSdf(lin);
        }
      }

      // Trilinear value.
      const f32 v00 = mps::lerpf(c[0], c[1], fx);
      const f32 v10 = mps::lerpf(c[2], c[3], fx);
      const f32 v01 = mps::lerpf(c[4], c[5], fx);
      const f32 v11 = mps::lerpf(c[6], c[7], fx);
      const f32 v0  = mps::lerpf(v00, v10, fy);
      const f32 v1  = mps::lerpf(v01, v11, fy);
      const f32 mu  = mps::lerpf(v0, v1, fz);

      // Analytic trilinear gradient.
      const f32 gx = mps::lerpf(mps::lerpf(c[1] - c[0], c[3] - c[2], fy),
                                mps::lerpf(c[5] - c[4], c[7] - c[6], fy), fz) / cellSize;
      const f32 gy = mps::lerpf(mps::lerpf(c[2] - c[0], c[3] - c[1], fx),
                                mps::lerpf(c[6] - c[4], c[7] - c[5], fx), fz) / cellSize;
      const f32 gz = mps::lerpf(mps::lerpf(c[4] - c[0], c[5] - c[1], fx),
                                mps::lerpf(c[6] - c[2], c[7] - c[3], fx), fy) / cellSize;

      return {mu, gx, gy, gz};
    }

    MPHD f32 eval(const vec3f& p) const { return evalGrad(p).x; }

    MPHD vec3f gradient(const vec3f& p) const {
      const vec4f f = evalGrad(p);
      return {f.y, f.z, f.w};
    }
  };

}
