#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

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

int main(int argc, char** argv) {
  mplog("mupsi2 renderer starting");

  // CLI: mupsi [scene] [width] [height]
  //   scene ∈ {sphere, spot, bunny, dragon}  (default: spot)
  const std::string scene = (argc > 1) ? argv[1] : "spot";
  const i32 w = (argc > 2) ? std::atoi(argv[2]) : 256;
  const i32 h = (argc > 3) ? std::atoi(argv[3]) : w;

  // 1. host staging
  mps::host::SceneConfig cfg;
  if      (scene == "sphere") cfg = mps::host::buildDefaultScene(w, h);
  else if (scene == "spot")   cfg = mps::host::buildSpotGpScene(w, h);
  else if (scene == "bunny")  cfg = mps::host::buildBunnyGpScene(w, h);
  else if (scene == "dragon") cfg = mps::host::buildDragonGpScene(w, h);
  else {
    std::cerr << "unknown scene '" << scene << "' (use sphere|spot|bunny|dragon)\n";
    return 1;
  }

  // 2. flatten to device POD arrays (CPU: plain heap; CUDA: cudaMalloc in the shell)
  mps::core::SceneDevice device = mps::host::toDevice(cfg.host);

  // 2b. bind a MeshMean medium to the flattened geometry (CPU = host pointers;
  //     the CUDA shell re-binds to device pointers after its own upload).
  if (cfg.medium.meanType == mps::core::MeanType::Mesh && !cfg.host.meshMeans.empty()) {
    mps::host::bindMeshMean(cfg.medium, device, cfg.host.meshMeans[0]);
  } else if (cfg.medium.meanType == mps::core::MeanType::Voxel) {
    mps::host::bindVoxelMean(cfg.medium, device);
  }

  mps::core::RenderContext ctx{device, cfg.medium, cfg.hasMedium, cfg.settings};

  // 3. framebuffer
  const i32 W = cfg.camera.width, H = cfg.camera.height;
  mps::FrameBuffer<mps::RGB_F32> fb;
  fb.width = W; fb.height = H;
  fb.pixels = new mps::RGB_F32[W * H];
  for (i32 i = 0; i < W * H; ++i) fb.pixels[i] = {0.0f, 0.0f, 0.0f};

  // 4. render (backend shell)
  auto t0 = std::chrono::steady_clock::now();
  mps::kernels::Entrance<ACTIVE_BACKEND>::render(ctx, cfg.camera, cfg.spp, fb);
  auto t1 = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  mplogtag("render", "{}x{} spp={} scene={}", W, H, cfg.spp, scene);
  std::cout << "render time: " << sec << " s\n";

  // 5. average + tonemap + save
  const f32 invSpp = 1.0f / f32(cfg.spp);
  mps::FrameBuffer<mps::RGB_8Bit> out;
  out.width = W; out.height = H;
  out.pixels = new mps::RGB_8Bit[W * H];
  for (i32 i = 0; i < W * H; ++i) fb.pixels[i] = fb.pixels[i] * invSpp;
  mps::host::framebuffer_radiance_to_8bit(fb, out);
  mps::host::save_ppm("test.ppm", out);

  delete[] fb.pixels;
  delete[] out.pixels;
  mps::host::freeDevice(device);

  mplog("done");
  return 0;
}
