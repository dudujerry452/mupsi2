#include "kernels/entrance/dispatch.h"

#include "core/math/random.h"

namespace mps::kernels {

  void Entrance<Backend::CPU>::render(const core::RenderContext& ctx, const core::Camera& cam,
                                      i32 spp, FrameBuffer<RGB_F32>& fb) {
    const i32 w = fb.width, h = fb.height;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (i32 y = 0; y < h; ++y) {
      for (i32 x = 0; x < w; ++x) {
        RGB_F32 acc{0.0f, 0.0f, 0.0f};
        for (i32 s = 0; s < spp; ++s) {
          u32 seed = core::make_seed(x, y, s, 42);
          core::Ray r = cam.generateRay(x, y);
          acc = acc + core::trace(r, ctx, seed);
        }
        fb.pixels[y * w + x] = acc;
      }
    }
  }

}
