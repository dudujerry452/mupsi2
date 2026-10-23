#pragma once

#include "core/basic/types.h"
#include "core/math/math_util.h"

namespace mps::core {

  // Three GPIS texturing approaches (test branches) + the heat-kernel (HKTEx).
  enum class ColorMethod : u16 { Constant = 0, Voxel = 1, Surface = 2, Kernel = 3, HeatKernel = 4 };

  // Dense RGB voxel grid (approach 1): bake a 3D color field, trilinear lookup.
  struct ColorGrid {
    vec3f origin{0.0f, 0.0f, 0.0f};
    f32   cellSize = 1.0f;
    u32   res = 0;
    const RGB_F32* data = nullptr;   // res^3, row-major x + res*(y + res*z)

    MPHD bool empty() const { return res == 0 || data == nullptr; }

    MPHD RGB_F32 eval(const vec3f& p) const {
      const vec3f g = (p - origin) / cellSize;
      const i32 ix = i32(mps::floorf(g.x)), iy = i32(mps::floorf(g.y)), iz = i32(mps::floorf(g.z));
      const f32 fx = g.x - f32(ix), fy = g.y - f32(iy), fz = g.z - f32(iz);
      auto idx = [&](i32 x, i32 y, i32 z) {
        x = mps::maxf(0, mps::minf(i32(res) - 1, x));
        y = mps::maxf(0, mps::minf(i32(res) - 1, y));
        z = mps::maxf(0, mps::minf(i32(res) - 1, z));
        return u32(x) + res * (u32(y) + res * u32(z));
      };
      const RGB_F32 c000 = data[idx(ix, iy, iz)],       c100 = data[idx(ix + 1, iy, iz)];
      const RGB_F32 c010 = data[idx(ix, iy + 1, iz)],   c110 = data[idx(ix + 1, iy + 1, iz)];
      const RGB_F32 c001 = data[idx(ix, iy, iz + 1)],   c101 = data[idx(ix + 1, iy, iz + 1)];
      const RGB_F32 c011 = data[idx(ix, iy + 1, iz + 1)], c111 = data[idx(ix + 1, iy + 1, iz + 1)];
      const RGB_F32 a = (c000 * (1.0f - fx) + c100 * fx) * (1.0f - fy) +
                        (c010 * (1.0f - fx) + c110 * fx) * fy;
      const RGB_F32 b = (c001 * (1.0f - fx) + c101 * fx) * (1.0f - fy) +
                        (c011 * (1.0f - fx) + c111 * fx) * fy;
      return a * (1.0f - fz) + b * fz;
    }
  };

  // 6-piece rainbow (hue sweep): fully saturated, full [0,1] value — used to make
  // the color schemes as vivid as possible. t ∈ [0,1] maps to one full hue cycle.
  MPHD MPINL RGB_F32 rainbow(f32 t) {
    t = mps::clampf(t, 0.0f, 1.0f);
    const f32 s = 6.0f * t;
    const i32 i = i32(mps::floorf(s));
    const f32 f = s - f32(i);
    switch (i) {
      case 0: return {1.0f, f, 0.0f};
      case 1: return {1.0f - f, 1.0f, 0.0f};
      case 2: return {0.0f, 1.0f, f};
      case 3: return {0.0f, 1.0f - f, 1.0f};
      case 4: return {f, 0.0f, 1.0f};
      default: return {1.0f, 0.0f, 1.0f - f};
    }
  }

  // Analytic 3D color field (approach 2, surface): a vivid rainbow sweeping along
  // the mesh's dominant horizontal axis, brightened vertically. Sampled directly
  // at the GP surface and baked into the voxel grid for approach 1.
  MPHD MPINL RGB_F32 fieldColor(const vec3f& p, const vec3f& mn, const vec3f& mx) {
    const vec3f d = mx - mn;
    const f32 tx = (d.x > 1e-6f) ? (p.x - mn.x) / d.x : 0.5f;
    const f32 ty = (d.y > 1e-6f) ? (p.y - mn.y) / d.y : 0.5f;
    const f32 tz = (d.z > 1e-6f) ? (p.z - mn.z) / d.z : 0.5f;
    const f32 hue = mps::clampf(0.7f * tx + 0.15f * ty + 0.15f * tz, 0.0f, 1.0f);
    const f32 val = 0.6f + 0.4f * ty;
    return rainbow(hue) * val;
  }

  // Spatial color kernel (approach 3): three feature points with pure primary
  // colors, blended by a sharp Gaussian over straight-line distance. The sharp
  // falloff keeps the three hues from greying out in between.
  MPHD MPINL RGB_F32 kernelColor(const vec3f& p, const vec3f& mn, const vec3f& mx) {
    const vec3f c = 0.5f * (mn + mx);
    const f32  s = 0.5f * mps::norm(mx - mn) + 1e-6f;
    const vec3f f0 = c + vec3f{-s, 0.0f, 0.0f};
    const vec3f f1 = c + vec3f{ s, 0.0f, 0.0f};
    const vec3f f2 = c + vec3f{0.0f, s, 0.0f};
    const vec3f f3 = c + vec3f{0.0f, -s, 0.0f};
    const f32 s2 = 0.10f * s * s;
    const f32 w0 = mps::expf(-mps::dot(p - f0, p - f0) / s2);
    const f32 w1 = mps::expf(-mps::dot(p - f1, p - f1) / s2);
    const f32 w2 = mps::expf(-mps::dot(p - f2, p - f2) / s2);
    const f32 w3 = mps::expf(-mps::dot(p - f3, p - f3) / s2);
    const f32 ws = w0 + w1 + w2 + w3 + 1e-6f;
    const RGB_F32 r = (RGB_F32{1.0f, 0.0f, 0.0f} * w0 +
                       RGB_F32{0.0f, 0.0f, 1.0f} * w1 +
                       RGB_F32{0.0f, 1.0f, 0.0f} * w2 +
                       RGB_F32{1.0f, 1.0f, 0.0f} * w3) / ws;
    return r;
  }

}
