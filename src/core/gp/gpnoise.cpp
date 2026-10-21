#include "core/gp/gpnoise.h"

#include "core/math/random.h"

namespace mps::core {

  MPHD vec4f SparseConvNoise::evalRaw(const vec3f& p, u32 seed) const {
    // Grid cell containing p (floor is the mathematically correct rounding).
    i32 cx = i32(mps::floorf(p.x / cellSize));
    i32 cy = i32(mps::floorf(p.y / cellSize));
    i32 cz = i32(mps::floorf(p.z / cellSize));

    vec4f acc{0.0f, 0.0f, 0.0f, 0.0f};

    // Sum impulses from the 3×3×3 neighborhood of cells covering the kernel's support.
    for (i32 dx = -1; dx <= 1; ++dx) {
      for (i32 dy = -1; dy <= 1; ++dy) {
        for (i32 dz = -1; dz <= 1; ++dz) {
          i32 nx = cx + dx, ny = cy + dy, nz = cz + dz;
          vec3f cellOrigin{f32(nx) * cellSize, f32(ny) * cellSize, f32(nz) * cellSize};

          for (u32 k = 0; k < pointsPerCell; ++k) {
            // Deterministic per-impulse random source.
            Random rng(make_seed(nx, ny, nz, k, seed));

            // Impulse position inside its cell, weight ±1 (unit variance).
            vec3f s = cellOrigin + vec3f{rng.next1D(), rng.next1D(), rng.next1D()} * cellSize;
            f32 w = (rng.next1D() < 0.5f) ? -1.0f : 1.0f;

            // Evaluate h once; ∇h = h·(-2/L²)(p-s).
            f32 L2 = mps::dot(kernel.lengthScale, kernel.lengthScale);
            f32 h = kernel.h(s, p);
            vec3f g = h * (-2.0f / L2) * (p - s);

            acc.x += w * h;
            acc.y += w * g.x;
            acc.z += w * g.y;
            acc.w += w * g.z;
          }
        }
      }
    }

    return acc;
  }

}
