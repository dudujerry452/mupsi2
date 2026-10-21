#pragma once

#include "core/basic/types.h"
#include "core/math/math_util.h"

namespace mps::core {

  // Dense voxel field of non-stationary kernel parameters {σ, L} (paper Eq. 25,
  // isotropic scaling: spatially-varying amplitude σ and length scale L).
  // Trilinear lookup; POD / device-friendly (pointers rebound by the backend).
  struct KernelField {
    vec3f origin{0.0f, 0.0f, 0.0f};
    f32   cellSize = 1.0f;
    u32   res = 0;
    const f32* sigma  = nullptr;   // res^3
    const f32* length = nullptr;   // res^3

    MPHD bool empty() const { return res == 0 || sigma == nullptr || length == nullptr; }

    // Trilinear {σ, L} at p (clamped to the grid bounds).
    MPHD vec2f eval(const vec3f& p) const {
      const vec3f g = (p - origin) / cellSize;
      const i32 ix = i32(mps::floorf(g.x));
      const i32 iy = i32(mps::floorf(g.y));
      const i32 iz = i32(mps::floorf(g.z));
      const f32 fx = g.x - f32(ix), fy = g.y - f32(iy), fz = g.z - f32(iz);

      auto idx = [&](i32 x, i32 y, i32 z) {
        x = mps::maxf(0, mps::minf(i32(res) - 1, x));
        y = mps::maxf(0, mps::minf(i32(res) - 1, y));
        z = mps::maxf(0, mps::minf(i32(res) - 1, z));
        return u32(x) + res * (u32(y) + res * u32(z));
      };
      auto tri = [&](const f32* a) {
        const f32 v00 = a[idx(ix, iy, iz)],     v10 = a[idx(ix + 1, iy, iz)];
        const f32 v01 = a[idx(ix, iy + 1, iz)], v11 = a[idx(ix + 1, iy + 1, iz)];
        const f32 v00z = a[idx(ix, iy, iz + 1)], v10z = a[idx(ix + 1, iy, iz + 1)];
        const f32 v01z = a[idx(ix, iy + 1, iz + 1)], v11z = a[idx(ix + 1, iy + 1, iz + 1)];
        const f32 a0 = mps::lerpf(mps::lerpf(v00, v10, fx), mps::lerpf(v01, v11, fx), fy);
        const f32 a1 = mps::lerpf(mps::lerpf(v00z, v10z, fx), mps::lerpf(v01z, v11z, fx), fy);
        return mps::lerpf(a0, a1, fz);
      };
      return {tri(sigma), tri(length)};
    }
  };

}
