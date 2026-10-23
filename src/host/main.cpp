#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "core/basic/core.h"
#include "core/bsdf/bsdf.h"
#include "core/utils/log.h"
#include "kernels/entrance/dispatch.h"
#include "host/scene_builder.h"
#include "host/data_manager.h"
#include "host/hdr_loader.h"
#include "host/heat_loader.h"
#include "host/ppm_util.h"

using namespace mps;

#if defined(MUPSI_USE_CUDA)
constexpr Backend ACTIVE_BACKEND = Backend::CUDA;
#else
constexpr Backend ACTIVE_BACKEND = Backend::CPU;
#endif

static void printUsage() {
  std::cout <<
    "mupsi2 — GPIS renderer\n"
    "\n"
    "Usage:\n"
    "  mupsi <scene> [w] [h] [mode] [bsdf] [spp] [sigma] [color] [heat] [--output FILE]\n"
    "  mupsi --scene <s> --width <w> --height <h> [--mode m] [--bsdf b] [--spp n]\n"
    "         [--sigma f] [--color c] [--heat FILE] [--output FILE]\n"
    "\n"
    "Options:\n"
    "  scene     sphere | spheregp | spot | bunny | dragon | dragonns   (default spot)\n"
    "  mode      single | renewal   (GP memory model override)\n"
    "  bsdf      lambertian | specular   (GP surface BSDF override)\n"
    "  spp       samples per pixel (default 1)\n"
    "  sigma     GP noise amplitude override\n"
    "  color     constant | voxel | surface | kernel | heatkernel   (texturing)\n"
    "  lscale    per-axis length scale Lx,Ly,Lz (comma-separated; unequal => anisotropic)\n"
    "  sigmafield x|y|z   ramp sigma monotonically along one axis (bottom -> top)\n"
    "  sigmarange min,max sigma at the two ends of the ramp (default 0.3,6.0)\n"
    "  sigmaclamp   step sigma at the model midpoint (t<0.5:min, t>=0.5:max)\n"
    "  margin  mx,my,mz   medium AABB margin as multiples of sigma (default 3,5,3)\n"
    "  voxelres N         voxel SDF grid resolution (default 256; higher = thinner features)\n"
    "  yaw/pitch orbit the camera AROUND the object (deg); 0/0 = default front view\n"
    "  heat      path to a precomputed HKTEx .bin (tools/precompute_heat.py)\n"
    "  output    output PPM path (default test.ppm)\n"
    "  -h/--help show this help\n";
}

int main(int argc, char** argv) {
  mplog("mupsi2 renderer starting");

  // ——— CLI parsing (positional for brevity, or --name value) ———
  std::string scene = "spot", modeArg, bsdfArg, colorArg, heatArg, outputArg = "test.ppm", lscaleArg, sigmaFieldArg, marginArg, sigmaRangeArg;
  i32 w = 256, h = 256, sppArg = 0, sigmaFieldAxis = -1, voxelResArg = 256;
  f32 sigmaArg = 0.0f, yawArg = 0.0f, pitchArg = 0.0f;
  f32 sigmaMin = 0.3f, sigmaMax = 6.0f;   // sigma-field ramp range
  bool sigmaClamp = false;                // step at model midpoint instead of ramp
  vec3f marginScale{3.0f, 5.0f, 3.0f};   // AABB margin (multiples of sigma)
  bool noMedia = false, noEnv = false;

  auto need = [&](i32& i, const char* name) -> std::string {
    if (i + 1 >= argc) { std::cerr << "missing value for " << name << "\n"; std::exit(1); }
    return argv[++i];
  };

  bool named = (argc > 1) && (argv[1][0] == '-');
  if (named) {
    for (i32 i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-h" || a == "--help") { printUsage(); return 0; }
      else if (a == "--scene")   scene = need(i, "--scene");
      else if (a == "--width")   w = std::atoi(need(i, "--width").c_str());
      else if (a == "--height")  h = std::atoi(need(i, "--height").c_str());
      else if (a == "--mode")    modeArg = need(i, "--mode");
      else if (a == "--bsdf")    bsdfArg = need(i, "--bsdf");
      else if (a == "--spp")     sppArg = std::atoi(need(i, "--spp").c_str());
      else if (a == "--sigma")   sigmaArg = std::atof(need(i, "--sigma").c_str());
      else if (a == "--color")   colorArg = need(i, "--color");
      else if (a == "--heat")    heatArg = need(i, "--heat");
      else if (a == "--lscale")  lscaleArg = need(i, "--lscale");
      else if (a == "--sigmafield") sigmaFieldArg = need(i, "--sigmafield");
      else if (a == "--sigmarange") sigmaRangeArg = need(i, "--sigmarange");
      else if (a == "--sigmaclamp") sigmaClamp = true;
      else if (a == "--margin")   marginArg = need(i, "--margin");
      else if (a == "--voxelres")  voxelResArg = std::atoi(need(i, "--voxelres").c_str());
      else if (a == "--output")  outputArg = need(i, "--output");
      else if (a == "--yaw")     yawArg = std::atof(need(i, "--yaw").c_str());
      else if (a == "--pitch")   pitchArg = std::atof(need(i, "--pitch").c_str());
      else if (a == "--nomedia") noMedia = true;
      else if (a == "--noenv")   noEnv = true;
      else { std::cerr << "unknown option '" << a << "'\n"; printUsage(); return 1; }
    }
  } else {
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
      printUsage(); return 0;
    }
    scene    = (argc > 1) ? argv[1] : "spot";
    w        = (argc > 2) ? std::atoi(argv[2]) : 256;
    h        = (argc > 3) ? std::atoi(argv[3]) : w;
    modeArg  = (argc > 4) ? argv[4] : "";
    bsdfArg  = (argc > 5) ? argv[5] : "";
    sppArg   = (argc > 6) ? std::atoi(argv[6]) : 0;
    sigmaArg = (argc > 7) ? std::atof(argv[7]) : 0.0f;
    colorArg = (argc > 8) ? argv[8] : "";
    heatArg  = (argc > 9) ? argv[9] : "";
    // extra args are ignored for positional brevity (use --output for the path)
  }

  // Sigma-field ramp range (absolute sigma at the two ends).
  if (!sigmaRangeArg.empty()) {
    if (std::sscanf(sigmaRangeArg.c_str(), "%f,%f", &sigmaMin, &sigmaMax) != 2 ||
        sigmaMin <= 0.0f || sigmaMax <= sigmaMin) {
      std::cerr << "bad --sigmarange '" << sigmaRangeArg << "' (need min,max with 0 < min < max)\n";
      return 1;
    }
  }

  // Medium AABB margin (multiples of sigma), per-axis.
  if (!marginArg.empty()) {
    f32 mx = 0, my = 0, mz = 0;
    if (std::sscanf(marginArg.c_str(), "%f,%f,%f", &mx, &my, &mz) != 3 ||
        mx <= 0.0f || my <= 0.0f || mz <= 0.0f) {
      std::cerr << "bad --margin '" << marginArg << "' (need mx,my,mz > 0)\n";
      return 1;
    }
    marginScale = {mx, my, mz};
  }

  // Sigma-field axis (spatially-varying sigma, ramp along one axis).
  if (!sigmaFieldArg.empty()) {
    if      (sigmaFieldArg == "x") sigmaFieldAxis = 0;
    else if (sigmaFieldArg == "y") sigmaFieldAxis = 1;
    else if (sigmaFieldArg == "z") sigmaFieldAxis = 2;
    else { std::cerr << "unknown --sigmafield '" << sigmaFieldArg << "' (use x|y|z)\n"; return 1; }
  }

  // 1. host staging. sigma must be known at build time (it sizes the medium
  //    bound margin AND the voxel narrow band), so fold the CLI override in here.
  const f32 meshSigma  = (sigmaArg > 0.0f) ? sigmaArg : 1.0f;
  const f32 sphereSigma = (sigmaArg > 0.0f) ? sigmaArg : 5.0f;

  mps::host::SceneConfig cfg;
  if      (scene == "sphere")   cfg = mps::host::buildDefaultScene(w, h);
  else if (scene == "spheregp") cfg = mps::host::buildGpSphereScene(w, h, sphereSigma);
  else if (scene == "spot")     cfg = mps::host::buildSpotGpScene(w, h, meshSigma, sigmaFieldAxis, sigmaMin, sigmaMax, sigmaClamp, marginScale, u32(voxelResArg));
  else if (scene == "bunny")    cfg = mps::host::buildBunnyGpScene(w, h, meshSigma, sigmaFieldAxis, sigmaMin, sigmaMax, sigmaClamp, marginScale, u32(voxelResArg));
  else if (scene == "dragon")   cfg = mps::host::buildDragonGpScene(w, h, meshSigma, sigmaFieldAxis, sigmaMin, sigmaMax, sigmaClamp, marginScale, u32(voxelResArg));
  else if (scene == "dragonns") // dragon + non-stationary kernel field (Eq. 25 test)
    cfg = mps::host::buildMeshGpScene("/home/dudujerry/models/dragon.obj", w, h,
                                      500.0f, 150.0f, core::GPMode::SingleRealization,
                                      meshSigma, 1.0f, true, 256, true, sigmaFieldAxis,
                                      sigmaMin, sigmaMax, sigmaClamp, marginScale);
  else {
    std::cerr << "unknown scene '" << scene << "' (use sphere|spheregp|spot|bunny|dragon|dragonns)\n";
    return 1;
  }

  // Orbit the camera AROUND the object (not just a view-direction change).
  // yaw=pitch=0 reproduces the original camera exactly. The orbit center is the
  // GP medium's AABB center; the radius is the original camera-to-center distance.
  if (yawArg != 0.0f || pitchArg != 0.0f) {
    const f32 yaw   = yawArg   * mps::pi / 180.0f;
    const f32 pitch = pitchArg * mps::pi / 180.0f;
    const vec3f objCenter = 0.5f * (cfg.medium.boundMin + cfg.medium.boundMax);
    const vec3f camHome   = cfg.camera.pos;
    vec3f dir = camHome - objCenter;      // home direction (yaw=pitch=0)
    const f32 R = mps::norm(dir);
    if (R < 1e-6f) dir = {0.0f, 0.0f, 1.0f};

    // yaw around world Y, then pitch around world X.
    const f32 cy = mps::cosf(yaw),  sy = mps::sinf(yaw);
    const f32 cp = mps::cosf(pitch), sp = mps::sinf(pitch);
    f32 x = dir.x * cy + dir.z * sy;
    f32 z = -dir.x * sy + dir.z * cy;
    f32 y = dir.y;
    f32 y2 = y * cp - z * sp;
    f32 z2 = y * sp + z * cp;
    dir = {x, y2, z2};

    const vec3f newPos = objCenter + dir;
    const vec3f forward = mps::normalize(objCenter - newPos);
    const vec3f up = (mps::fabsf(forward.y) > 0.99f) ? vec3f{0.0f, 0.0f, 1.0f}
                                                     : vec3f{0.0f, 1.0f, 0.0f};
    cfg.camera = mps::core::makeCamera(newPos, forward, up, 45.0f, w, h);
  }

  // Debug: disable the GP medium (render hard surfaces only).
  if (noMedia) cfg.hasMedium = false;

  // Override the GP memory model (for the 7.1 test matrix).
  if (modeArg == "single")       cfg.medium.gpMode = core::GPMode::SingleRealization;
  else if (modeArg == "renewal") cfg.medium.gpMode = core::GPMode::RenewalPlus;
  else if (!modeArg.empty()) {
    std::cerr << "unknown mode '" << modeArg << "' (use single|renewal)\n";
    return 1;
  }

  // Override the GP surface BSDF (Lambertian / specular microsurface).
  if (bsdfArg == "lambertian") cfg.medium.surfaceBsdf = u16(core::BsdfType::Lambertian);
  else if (bsdfArg == "specular") cfg.medium.surfaceBsdf = u16(core::BsdfType::Specular);
  else if (!bsdfArg.empty()) {
    std::cerr << "unknown bsdf '" << bsdfArg << "' (use lambertian|specular)\n";
    return 1;
  }

  if (sppArg > 0) cfg.spp = sppArg;

  // Per-axis length scale override (anisotropic kernel when components differ).
  if (!lscaleArg.empty()) {
    f32 lx = 0, ly = 0, lz = 0;
    if (std::sscanf(lscaleArg.c_str(), "%f,%f,%f", &lx, &ly, &lz) != 3 ||
        lx <= 0.0f || ly <= 0.0f || lz <= 0.0f) {
      std::cerr << "bad --lscale '" << lscaleArg << "' (need Lx,Ly,Lz, each > 0)\n";
      return 1;
    }
    cfg.medium.lengthScale = {lx, ly, lz};
  }

  // GPIS texturing approach.
  if (colorArg == "constant")    cfg.medium.colorMethod = u16(core::ColorMethod::Constant);
  else if (colorArg == "voxel")  cfg.medium.colorMethod = u16(core::ColorMethod::Voxel);
  else if (colorArg == "surface") cfg.medium.colorMethod = u16(core::ColorMethod::Surface);
  else if (colorArg == "kernel")  cfg.medium.colorMethod = u16(core::ColorMethod::Kernel);
  else if (colorArg == "heatkernel") cfg.medium.colorMethod = u16(core::ColorMethod::HeatKernel);
  else if (!colorArg.empty()) {
    std::cerr << "unknown color '" << colorArg << "' (use voxel|surface|kernel|heatkernel)\n";
    return 1;
  }

  // Load a precomputed HKTEx heat-kernel texture (tools/precompute_heat.py).
  if (!heatArg.empty()) {
    if (mps::host::loadHeatKernel(heatArg, cfg.host)) {
      cfg.medium.colorMethod = u16(core::ColorMethod::HeatKernel);
    }
    else
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
  if (cfg.medium.kernelField.res > 0) {
    mps::host::bindKernelField(cfg.medium, device);
  }
  if (cfg.medium.colorGrid.res > 0) {
    mps::host::bindColorGrid(cfg.medium, device);
  }
  if (cfg.host.heatN > 0) {
    mps::host::bindHeatKernel(cfg.medium, device);
    if (!cfg.host.meshMeans.empty()) {
      cfg.medium.heatKernel.bvhRoot       = cfg.host.meshMeans[0].bvhRoot;
      cfg.medium.heatKernel.bvhNodeOffset = cfg.host.meshMeans[0].bvhNodeOffset;
      cfg.medium.heatKernel.bvhTriOffset  = cfg.host.meshMeans[0].bvhTriOffset;
    }
  }

  mps::core::RenderContext ctx{device, cfg.medium, cfg.hasMedium, cfg.settings};

  // Environment map (background/IBL).
  mps::RGB_F32* envPixels = nullptr;
  {
    mps::host::HdrImage env = mps::host::loadHdr("/home/dudujerry/models/envmap.hdr");
    if (!noEnv && env.width > 0 && env.height > 0) {
      // Scale the HDR down so indirect/envmap reflections don't blow out the
      // render (the raw sky is ~1-10+ radiance).
      constexpr f32 ENVMAP_SCALE = 0.15f;
      envPixels = new mps::RGB_F32[env.pixels.size()];
      for (size_t i = 0; i < env.pixels.size(); ++i)
        envPixels[i] = env.pixels[i] * ENVMAP_SCALE;
      ctx.envMap.width  = env.width;
      ctx.envMap.height = env.height;
      ctx.envMap.pixels = envPixels;
    }
  }

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
  mps::host::save_ppm(outputArg, out);

  delete[] fb.pixels;
  delete[] out.pixels;
  delete[] envPixels;
  mps::host::freeDevice(device);

  mplog("done");
  return 0;
}
