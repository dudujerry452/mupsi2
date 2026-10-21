#include <chrono>
#include <iostream>

#include "core/basic/core.h"
#include "core/utils/log.h"
#include "kernels/entrance/dispatch.h"
#include "host/scene_builder.h"
#include "host/data_manager.h"
#include "host/ppm_util.h"

using namespace mps;

#if defined(MUPSI_USE_CUDA)
constexpr Backend ACTIVE_BACKEND = Backend::CUDA;
#else
constexpr Backend ACTIVE_BACKEND = Backend::CPU;
#endif

int main() {
  mplog("mupsi2 renderer starting");

  // 1. host staging
  auto cfg = mps::host::buildDefaultScene();

  // 2. flatten to device POD arrays (CPU: plain heap; CUDA: would cudaMalloc)
  mps::core::SceneDevice device = mps::host::toDevice(cfg.host);
  mps::core::RenderContext ctx{device, cfg.medium, cfg.hasMedium, cfg.settings};

  // 3. framebuffer
  const i32 w = cfg.camera.width, h = cfg.camera.height;
  mps::FrameBuffer<mps::RGB_F32> fb;
  fb.width = w; fb.height = h;
  fb.pixels = new mps::RGB_F32[w * h];
  for (i32 i = 0; i < w * h; ++i) fb.pixels[i] = {0.0f, 0.0f, 0.0f};

  // 4. render (backend shell)
  auto t0 = std::chrono::steady_clock::now();
  mps::kernels::Entrance<ACTIVE_BACKEND>::render(ctx, cfg.camera, cfg.spp, fb);
  auto t1 = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  mplogtag("render", "{}x{} spp={}", w, h, cfg.spp);
  std::cout << "render time: " << sec << " s\n";

  // 5. average + tonemap + save
  const f32 invSpp = 1.0f / f32(cfg.spp);
  mps::FrameBuffer<mps::RGB_8Bit> out;
  out.width = w; out.height = h;
  out.pixels = new mps::RGB_8Bit[w * h];
  for (i32 i = 0; i < w * h; ++i) fb.pixels[i] = fb.pixels[i] * invSpp;
  mps::host::framebuffer_radiance_to_8bit(fb, out);
  mps::host::save_ppm("test.ppm", out);

  delete[] fb.pixels;
  delete[] out.pixels;
  mps::host::freeDevice(device);

  mplog("done");
  return 0;
}
