#include "host/voxel_builder.h"

#include <cmath>

#include "core/basic/core.h"
#include "core/utils/log.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mps::host {

  VoxelGridHost buildSparseSDFGrid(const core::MeshMean& mm, const core::AABB& bounds,
                                   u32 res, f32 bandCells) {
    VoxelGridHost grid;

    if (res == 0 || mm.empty() || !bounds.isValid()) {
      mplog("buildSparseSDFGrid: invalid input (res={} meshEmpty={})", res, mm.empty());
      return grid;
    }

    const vec3f diag = bounds.diagonal();
    const f32 maxExtent = mps::maxf(diag.x, mps::maxf(diag.y, diag.z));
    grid.cellSize = maxExtent / f32(res);
    grid.band     = bandCells * grid.cellSize;
    grid.res      = res;

    // Pad so the whole narrow band sits inside the grid, plus a little exterior.
    const f32 margin = grid.band + 1.5f * grid.cellSize;
    grid.origin = bounds.min - vec3f{margin, margin, margin};

    const u64 total = u64(res) * u64(res) * u64(res);
    grid.signGrid.assign(size_t(total), 0);

    // Temporary full-precision SDF so the band extraction is a single ordered
    // pass (no second BVH query round).
    std::vector<f32> sdf(static_cast<size_t>(total));

    mplog("buildSparseSDFGrid: res={} cells={} cellSize={:.3f} band={:.3f} ({} nodes)",
          res, total, grid.cellSize, grid.band, total);

    const u32 R = res;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i64 lin = 0; lin < i64(total); ++lin) {
      const u32 ix = u32(lin % R);
      const u32 iy = u32((lin / R) % R);
      const u32 iz = u32(lin / (i64(R) * R));
      const vec3f nodePos = grid.origin + vec3f{f32(ix), f32(iy), f32(iz)} * grid.cellSize;

      const f32 d = mm.eval(nodePos);          // signed distance (pseudo-normal sign)
      sdf[size_t(lin)] = d;
      grid.signGrid[size_t(lin)] = (d > 0.0f) ? u8(1) : u8(0);
    }

    // Ordered narrow-band extraction (bandIndex stays sorted by construction).
    grid.bandIndex.reserve(size_t(total) / 8);
    grid.bandValue.reserve(size_t(total) / 8);
    for (u64 lin = 0; lin < total; ++lin) {
      const f32 d = sdf[size_t(lin)];
      if (mps::fabsf(d) < grid.band) {
        grid.bandIndex.push_back(u32(lin));
        grid.bandValue.push_back(d);
      }
    }

    mplog("buildSparseSDFGrid: narrow band kept {}/{} nodes ({:.1f}%)",
          grid.bandIndex.size(), total, 100.0 * double(grid.bandIndex.size()) / double(total));

    return grid;
  }

}
