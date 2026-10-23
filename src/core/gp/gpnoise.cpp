#include "core/gp/gpnoise.h"

#include "core/math/random.h"

namespace mps::core {

  MPHD vec4f SparseConvNoise::evalRaw(const vec3f& p, u32 seed) const {
    // Grid cell containing p (floor is the mathematically correct rounding).
    i32 cx = i32(mps::floorf(p.x / cellSize));
    i32 cy = i32(mps::floorf(p.y / cellSize));
    i32 cz = i32(mps::floorf(p.z / cellSize));

    const vec3f L2 = lengthScale2At(p);   // per-axis {Lx², Ly², Lz²}
    const f32 support2 = cellSize * cellSize;   // compact support radius = cellSize (= kernelRadius)

    // Adaptive neighborhood: the kernel's support extends Li·cellSize along axis
    // i (the anisotropic-distance cutoff q = support2), so iterate ceil(Li) cells
    // each way. Isotropic default Li=1 → the original 3×3×3 (±1 cell) block.
    auto toCell = [&](f32 L) -> i32 {   // ceil(L) without a math call (device-safe)
      const i32 f = i32(mps::floorf(L));
      return (L > f32(f)) ? f + 1 : f;
    };
    const i32 rx = mps::maxf(1, toCell(mps::sqrtf(L2.x)));
    const i32 ry = mps::maxf(1, toCell(mps::sqrtf(L2.y)));
    const i32 rz = mps::maxf(1, toCell(mps::sqrtf(L2.z)));

    vec4f acc{0.0f, 0.0f, 0.0f, 0.0f};

    for (i32 dx = -rx; dx <= rx; ++dx) {
      for (i32 dy = -ry; dy <= ry; ++dy) {
        for (i32 dz = -rz; dz <= rz; ++dz) {
          i32 nx = cx + dx, ny = cy + dy, nz = cz + dz;
          vec3f cellOrigin{f32(nx) * cellSize, f32(ny) * cellSize, f32(nz) * cellSize};

          // One hash-seeded RNG per cell; impulses drawn sequentially (no warmup).
          // Matches paper Algorithm 1: rng <- RNG(HASH(seed, cellIndex + xyz)).
          Random rng(make_seed(nx, ny, nz, seed));
          for (u32 k = 0; k < pointsPerCell; ++k) {
            // Impulse position inside its cell, weight ±1 (unit variance).
            vec3f s = cellOrigin + vec3f{rng.next1D(), rng.next1D(), rng.next1D()} * cellSize;
            f32 w = (rng.next1D() < 0.5f) ? -1.0f : 1.0f;

            // Compact-support cull in the ANISOTROPIC distance (paper §4.2.2,
            // reference cell3D): skip impulses with q ≥ support² without evaluating
            // exp().
            vec3f d = p - s;
            const f32 q = (d.x * d.x) / L2.x + (d.y * d.y) / L2.y + (d.z * d.z) / L2.z;
            if (q >= support2) continue;

            // Evaluate h once; ∇h = h·(-2)(d_x/Lx², d_y/Ly², d_z/Lz²).
            f32 h = mps::expf(-q);
            vec3f g = h * vec3f{-2.0f * d.x / L2.x, -2.0f * d.y / L2.y, -2.0f * d.z / L2.z};

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

  MPHD vec2f SparseConvNoise::eval1D(const vec3f& p, f32 t, u32 seed) const {
    // Grid cell containing t along the ray.
    const f32 tGrid = t / cellSize;
    const f32 frac  = tGrid - mps::floorf(tGrid);
    const i32 ci    = i32(mps::floorf(tGrid));

    const f32 L2       = lengthScale2At(p).x;   // isotropic scalar (1D path only)
    const f32 support2 = cellSize * cellSize;   // compact support radius = cellSize

    f32 val = 0.0f, gradT = 0.0f;

    // 3 cells along the ray (dx = -1, 0, 1) — the 1D analogue of the 3×3×3
    // neighborhood. Matches paper §7 (3^d → 3^1) and the reference cell1D.
    for (i32 dx = -1; dx <= 1; ++dx) {
      const i32 cellIdx = ci + dx;
      Random rng(make_seed(cellIdx, seed));
      for (u32 k = 0; k < pointsPerCell; ++k) {
        const f32 ti = rng.next1D();                                   // impulse position in cell
        const f32 w  = (rng.next1D() < 0.5f) ? -1.0f : 1.0f;
        const f32 dt = (frac - f32(dx) - ti) * cellSize;               // world distance along ray

        if (dt * dt >= support2) continue;                             // compact-support cull

        const f32 h = mps::expf(-(dt * dt) / L2);
        val   += w * h;
        gradT += w * h * (-2.0f / L2) * dt;
      }
    }

    const f32 scale = sigmaAt(p) / mps::sqrtf(variance1D(p));
    return {val * scale, gradT * scale};
  }

}
