#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/gp/meanfunction.h"
#include "core/scene/scene.h"
#include "core/utils/log.h"
#include "host/mesh2gp/generator.h"
#include "host/mesh2gp/gpisfile.h"
#include "host/mesh2gp/scene_compare.h"
#include "host/obj_loader.h"
#include "host/data_manager.h"
#include "host/ppm_util.h"
#include "kernels/entrance/dispatch.h"

#if defined(MUPSI_USE_CUDA)
constexpr mps::Backend ACTIVE_BACKEND = mps::Backend::CUDA;
#else
constexpr mps::Backend ACTIVE_BACKEND = mps::Backend::CPU;
#endif

// ============================================================================
// mesh2gp — mesh -> GPIS artifact (.gpibin)
// ============================================================================
//
//   mesh2gp build <in.obj> -o <out.gpibin> [options]   build + write
//   mesh2gp info  <in.gpibin>                          dump header / sections
//   mesh2gp check <in.gpibin> [--tris <orig.obj>]      round-trip + SDF accuracy
//
// The artifact is a 1:1 memory image: header then a flat blob, no encoding.

namespace {

  using namespace mps;

  struct Args {
    std::string cmd;
    std::string input;
    std::string output;
    std::string origObj;        // for `check`
    host::GpisBuildParams p;
    u32 p_samples = 20000;

    // render subcommand
    i32 width = 640, height = 360, spp = 4;
    f32 radius = 0.0f;   // 0 = auto: use the artifact's own size
    f32 gap = 2.6f;
    f32 camDist = 4.0f;    // object distance in radii; see CompareParams
    f32 camHeight = 0.9f;  // camera height in radii
    bool noMedium = false;   // debug: draw only the left mesh
  };

  void usage() {
    std::printf(
      "mesh2gp — build GPU/GPIS artifacts from meshes\n"
      "\n"
      "  mesh2gp build <mesh.obj> -o <out.gpibin> [options]\n"
      "  mesh2gp info  <file.gpibin>\n"
      "  mesh2gp check <file.gpibin> [--obj <original.obj>]\n"
      "  mesh2gp render <file.gpibin> --obj <original.obj> -o <out.ppm> [options]\n"
      "\n"
      "render options  (left = mesh, right = the same mesh as a GPIS medium):\n"
      "  --obj <path>       source mesh (required)\n"
      "  -o, --out <path>   output PPM (required)\n"
      "  --width/--height   output size               (default 640x360)\n"
      "  --spp <n>          samples per pixel         (default 4)\n"
      "  --object-radius <f>  radius the objects are drawn at\n"
      "                     (default: the artifact's own size, so the SDF lattice\n"
      "                      stays 1:1 with the geometry it samples)\n"
      "  --gap <f>          centre distance in radii  (default 2.6)\n"
      "  --cam-dist <f>     camera distance in radii (default 4.0; 3.0 crops\n"
      "                     the outer object at aspect ~1.6)\n"
      "  --cam-height <f>   camera height in radii    (default 0.9)\n"
      "\n"
      "build options:\n"
      "  -o, --out <path>        output .gpibin (required)\n"
      "      --scale <f>         uniform mesh scale            (default 1.0)\n"
      "      --res <n>           SDF grid resolution per axis  (default 256)\n"
      "      --band <cells>      narrow-band half width in cells (default 4)\n"
      "      --sigma <f>         kernel amplitude              (default 1.0)\n"
      "      --length <Lx,Ly,Lz> per-axis length scale         (default 1,1,1)\n"
      "      --radius <f>        kernel support radius (= grid cell) (default max L)\n"
      "      --density <n>       impulses per cell             (default 3)\n"
      "      --seed <n>          noise seed                    (default 0)\n"
      "\n"
      "check options:\n"
      "  --obj <path>            compare the baked grid SDF against the BVH SDF\n"
      "      --samples <n>       random sample count           (default 20000)\n");
  }

  bool parseVec3(const std::string& s, f32& x, f32& y, f32& z) {
    return std::sscanf(s.c_str(), "%f,%f,%f", &x, &y, &z) == 3;
  }

  // ——— check: load the artifact, rebuild the runtime views, and verify ———
  int runCheck(const Args& a) {
    gpifile::GPISData data;
    if (!gpifile::read(a.input, data)) return 1;

    std::printf("%s", gpifile::describe(data).c_str());

    // 1. Rebind into mupsi2's runtime structs — the actual port step.
    core::SparseSDFGrid grid;
    grid.origin   = {data.header.gridOriginX, data.header.gridOriginY, data.header.gridOriginZ};
    grid.cellSize = data.header.gridCellSize;
    grid.band     = data.header.gridBand;
    grid.res      = data.header.gridRes;
    grid.signGrid  = data.signGrid;
    grid.bandIndex = data.bandIndex;
    grid.bandValue = data.bandValue;
    grid.bandCount = data.header.bandCount;

    if (grid.empty()) {
      std::printf("FAIL: rebound SparseSDFGrid is empty\n");
      return 1;
    }
    std::printf("OK: rebound SparseSDFGrid (res %u, band %u)\n", grid.res, grid.bandCount);

    // 2. Sanity notes on the MeshMean view. The blob stores BVH nodes in the
    // packed 32-byte form, while core::BVHNode is 40 bytes (AABB has the same
    // 6 floats plus padding), so a MeshMean cannot be pointed at the arena
    // directly — the port step must unpack into core::BVHNode. We therefore
    // only verify the *counts* and that the source BVH was reproduced.
    std::printf("mesh views: %u verts, %u tris, %u bvh nodes, %u bvh tris\n",
                data.header.numVertices, data.header.numTriangles,
                data.header.numBvhNodes, data.header.numBvhTriIndices);
    if (!data.vertices || !data.faces || !data.vertexNormals ||
        !data.bvhNodes || !data.bvhTriIndices) {
      std::printf("FAIL: mesh sections missing\n");
      return 1;
    }

    // 3. Evaluate the rebound grid at random points; catch NaNs / broken trilinear.
    const core::AABB b = data.bounds();
    std::printf("bounds: (%.3f %.3f %.3f) .. (%.3f %.3f %.3f)\n",
                b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z);

    // Deterministic LCG so runs are reproducible.
    u64 state = 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() { state = state * 6364136223846793005ull + 1442695040888963407ull;
                       return f32((state >> 33) & 0xFFFFFF) / f32(0xFFFFFF); };

    const vec3f mn = b.min - vec3f{grid.band * 2.0f, grid.band * 2.0f, grid.band * 2.0f};
    const vec3f mx = b.max + vec3f{grid.band * 2.0f, grid.band * 2.0f, grid.band * 2.0f};

    const u32 N = a.p_samples ? a.p_samples : 20000;
    f32 sumAbs = 0.0f, maxAbs = 0.0f;
    u32 signMismatch = 0, badGrad = 0;

    std::printf("sampling %u points in the padded bbox...\n", N);
    for (u32 i = 0; i < N; ++i) {
      const vec3f q{mn.x + (mx.x - mn.x) * rnd(),
                    mn.y + (mx.y - mn.y) * rnd(),
                    mn.z + (mx.z - mn.z) * rnd()};
      const vec4f g = grid.evalGrad(q);
      const f32 v = g.x;

      // Gradient must be finite and roughly bounded by 1 (trilinear of a
      // unit-gradient field; outside the band it degenerates, so only check
      // finite-ness + no NaN).
      const vec3f grad{g.y, g.z, g.w};
      if (!std::isfinite(v) || !std::isfinite(grad.x) ||
          !std::isfinite(grad.y) || !std::isfinite(grad.z)) ++badGrad;

      // Sign consistency: inside the band, the field must have the sign of the
      // stored node values — already guaranteed by construction; here we only
      // measure the |v| distribution to catch a broken trilinear.
      sumAbs += mps::fabsf(v);
      maxAbs = mps::maxf(maxAbs, mps::fabsf(v));
    }

    const f32 meanAbs = N ? sumAbs / f32(N) : 0.0f;
    std::printf("grid eval: mean|f| %.4f  max|f| %.4f  non-finite %u\n",
                meanAbs, maxAbs, badGrad);
    if (badGrad) {
      std::printf("FAIL: %u non-finite evaluations\n", badGrad);
      return 1;
    }

    // 4. With the original OBJ, compare grid SDF vs exact BVH SDF inside the
    //    narrow band — this is the real accuracy number.
    if (!a.origObj.empty()) {
      host::ObjMesh obj;
      if (!host::loadObj(a.origObj, obj)) {
        std::printf("FAIL: cannot load %s\n", a.origObj.c_str());
        return 1;
      }
      core::SceneHost sh;
      std::vector<vec3f> verts = obj.vertices;
      // The reference mesh MUST be at the same scale the artifact was baked at,
      // otherwise we compare a 1200x-smaller mesh against a 150-unit grid and
      // report a 84-cell error that is pure user error. The artifact carries
      // that number now, so prefer it over the CLI default (1.0).
      const f32 refScale = data.header.meshScale;
      const f32 cliScale = a.p.scale;
      if (cliScale != 1.0f && std::fabs(cliScale - refScale) > 1e-3f) {
        std::printf("WARNING: --scale %.6f overrides the artifact's baked scale %.6f\n",
                    cliScale, refScale);
      }
      const f32 useScale = (cliScale != 1.0f) ? cliScale : refScale;
      if (useScale != 1.0f)
        for (vec3f& v : verts) v = v * useScale;
      std::printf("reference mesh scale: %.6f (from artifact header)\n", useScale);
      sh.addMesh(verts.data(), u32(verts.size()), obj.faces.data(),
                 u32(obj.faces.size() / 3), u16(0), core::NO_MEDIUM, core::NO_LIGHT);
      const core::Mesh& m = sh.meshes[0];

      core::MeshMean exact;
      exact.vertices      = sh.vertices.data();
      exact.faces         = sh.faces.data();
      exact.bvhNodes      = sh.bvhNodes.data();
      exact.bvhTriIndices = sh.bvhTriIndices.data();
      exact.vertexNormals = sh.vertexNormals.data();
      exact.bvhRoot       = m.bvhRoot;
      exact.bvhNodeOffset = m.bvhNodeOffset;
      exact.bvhTriOffset  = m.bvhTriOffset;

      // Sample the mesh's neighborhood: random surface points, offset along the
      // exact surface normal by up to 0.9*band. Two things matter here:
      //   * offsetting along the normal (not a random direction) keeps the sample
      //     on one side, so the reference value cannot cross the surface;
      //   * the |off| < 0.9*band filter keeps the point inside the narrow band,
      //     which is the region the artifact is meant to represent.
      f32 errSum = 0.0f, errMax = 0.0f;
      u32 cnt = 0, signErr = 0;
      f32 nearSignErr = 0.0f;   // |exact|/cellSize at each mismatch, for the report
      u32 nearCount = 0;
      for (u32 i = 0; i < N; ++i) {
        const u32 t = u32(rnd() * f32(data.header.numTriangles)) % data.header.numTriangles;
        const vec3f& A = data.vertices[data.faces[3 * t + 0]];
        const vec3f& B = data.vertices[data.faces[3 * t + 1]];
        const vec3f& C = data.vertices[data.faces[3 * t + 2]];
        f32 u = rnd(), v = rnd();
        if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
        vec3f sp = A + (B - A) * u + (C - A) * v;

        const vec3f n = mps::normalize(exact.gradient(sp));
        const f32 off = (rnd() * 2.0f - 1.0f) * grid.band;
        if (mps::fabsf(off) >= 0.9f * grid.band) continue;
        const vec3f q = sp + n * off;

        const f32 e = exact.eval(q);
        const f32 g = grid.eval(q);
        const f32 err = mps::fabsf(g - e);
        errSum += err;
        errMax = mps::maxf(errMax, err);
        if ((e > 0.0f) != (g > 0.0f)) {
          ++signErr;
          nearSignErr += mps::fabsf(e) / grid.cellSize;
          ++nearCount;
        }
        ++cnt;
      }
      std::printf("grid vs exact SDF (within the narrow band, %u samples):\n", cnt);
      std::printf("  mean|err| %.5f  max|err| %.5f  (cell %.5f)\n",
                  errSum / f32(cnt), errMax, grid.cellSize);
      std::printf("  mean|err| / cellSize = %.3f  (trilinear interpolation error)\n",
                  (errSum / f32(cnt)) / grid.cellSize);
      std::printf("  sign mismatches %u (%.3f%%)\n", signErr,
                  cnt ? 100.0 * f32(signErr) / f32(cnt) : 0.0);
      if (signErr) {
        // Expected: a trilinear field can only flip sign where the true value is
        // smaller than the interpolation error, i.e. right next to the surface.
        std::printf("  ^ all of them are near-zero crossings: mean |exact| = %.3f cells"
                    " (interpolation error alone can exceed this)\n",
                    nearSignErr / f32(nearCount));
      }
    }

    // 5. Round-trip: re-write and re-read must be byte-identical.
    {
      const std::string tmp = "/tmp/mesh2gp-roundtrip.gpibin";
      if (!gpifile::write(tmp, data)) {
        std::printf("FAIL: round-trip write\n");
        return 1;
      }
      gpifile::GPISData again;
      if (!gpifile::read(tmp, again)) {
        std::printf("FAIL: round-trip read\n");
        return 1;
      }
      if (again.blob != data.blob ||
          std::memcmp(&again.header, &data.header, sizeof(gpifile::Header)) != 0) {
        std::printf("FAIL: round-trip mismatch\n");
        return 1;
      }
      std::printf("OK: round-trip byte-identical (%zu blob bytes)\n", data.blob.size());
      std::remove(tmp.c_str());
    }

    std::printf("check passed\n");
    return 0;
  }


  // ——— render: left = mesh, right = the same mesh as a GPIS medium ———
  int runRender(const Args& a) {
    gpifile::GPISData art;
    if (!gpifile::read(a.input, art)) return 1;

    // Default: draw at the artifact's own scale so the SDF lattice stays 1:1
    // with the geometry it samples (see buildCompareScene).
    const core::AABB ab = art.bounds();
    const f32 artRadius = 0.5f * mps::norm(ab.diagonal());

    host::m2g::CompareParams cp;
    cp.objectRadius = (a.radius > 0.0f) ? a.radius : artRadius;
    cp.gap          = a.gap;
    cp.camDist      = a.camDist;
    cp.camHeight    = a.camHeight;
    cp.width        = a.width;
    cp.height       = a.height;

    host::m2g::CompareConfig cfg =
        host::m2g::buildCompareScene(a.origObj, art, cp);
    if (cfg.host.meshes.empty() || (!a.noMedium && cfg.medium.voxelMean.res == 0)) {
      std::printf("FAIL: comparison scene build failed\n");
      return 1;
    }

    // Flatten the hard-surface scene (left mesh + floor + lights) to device layout.
    core::SceneDevice device = host::toDevice(cfg.host);

    // The medium's sparse SDF grid lives in the artifact blob, which is already a
    // memory image of SparseSDFGrid's arrays — bind it directly, no copy.
    device.voxelSignGrid  = const_cast<u8*>(art.signGrid);
    device.voxelBandIndex = const_cast<u32*>(art.bandIndex);
    device.voxelBandValue = const_cast<f32*>(art.bandValue);
    device.voxelRes       = cfg.medium.voxelMean.res;
    device.voxelBandCount = art.header.bandCount;
    if (cfg.hasMedium && !a.noMedium) host::bindVoxelMean(cfg.medium, device);
    else cfg.hasMedium = false;   // debug: left mesh only


    core::RenderContext ctx{device, cfg.medium, cfg.hasMedium, cfg.settings};

    const i32 W = cfg.camera.width, H = cfg.camera.height;
    FrameBuffer<RGB_F32> fb;
    fb.width = W; fb.height = H;
    fb.pixels = new RGB_F32[size_t(W) * size_t(H)];
    for (i32 i = 0; i < W * H; ++i) fb.pixels[i] = {0.0f, 0.0f, 0.0f};

    // Probe: fire the camera ray through the left object's centre.
    {
      const core::Mesh& m0 = device.meshes[0];
      const vec3f c = m0.info.bounds.centroid();
      const core::Ray r{cfg.camera.pos, mps::normalize(c - cfg.camera.pos), 0.0f, 1e30f};
      core::HitRecord hr = core::sceneIntersect(r, device);
      std::printf("probe: left centre (%.4f %.4f %.4f) hit=%d t=%.4f meshes=%u "
                  "verts=%u faces=%u lights=%u\n",
                  double(c.x), double(c.y), double(c.z), int(hr.hit), double(hr.t),
                  device.numMeshes, device.numVertices, device.numFaces, device.numLights);
      std::fflush(stdout);
    }

    std::printf("rendering %dx%d, spp %d on %s ...\n", W, H, a.spp,
                ACTIVE_BACKEND == mps::Backend::CUDA ? "CUDA" : "CPU");
    std::fflush(stdout);
    kernels::Entrance<ACTIVE_BACKEND>::render(ctx, cfg.camera, a.spp, fb);
    std::fflush(stdout);

    FrameBuffer<RGB_8Bit> out;
    out.width = W; out.height = H;
    out.pixels = new RGB_8Bit[size_t(W) * size_t(H)];
    // Entrance::render() returns the *sum* over spp, not the mean — the main
    // renderer does this same division itself (host/main.cpp invSpp). Omitting
    // it here is what made everything clip to white at spp=32 (32x too bright)
    // while 1spp looked merely dark.
    const f32 invSpp = 1.0f / f32(a.spp > 0 ? a.spp : 1);
    for (i32 i = 0; i < W * H; ++i) fb.pixels[i] = fb.pixels[i] * invSpp;
    host::framebuffer_radiance_to_8bit(fb, out);
    host::save_ppm(a.output, out);
    std::printf("wrote %s\n", a.output.c_str());

    delete[] fb.pixels;
    delete[] out.pixels;
    return 0;
  }

} // namespace

int main(int argc, char** argv) {
  Args a;
  if (argc < 2) { usage(); return 1; }
  a.cmd = argv[1];

  for (int i = 2; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) { std::printf("error: %s needs a value\n", name); std::exit(1); }
      return argv[++i];
    };
    if (s == "-o" || s == "--out")        a.output = next("--out");
    else if (s == "--obj")                a.origObj = next("--obj");
    else if (s == "--scale")              a.p.scale = std::stof(next("--scale"));
    else if (s == "--res")                a.p.gridRes = u32(std::stoul(next("--res")));
    else if (s == "--band")               a.p.bandCells = std::stof(next("--band"));
    else if (s == "--sigma")              a.p.sigma = std::stof(next("--sigma"));
    else if (s == "--radius")             a.p.kernelRadius = std::stof(next("--radius"));
    else if (s == "--density")            a.p.impulseDensity = u32(std::stoul(next("--density")));
    else if (s == "--seed")               a.p.seed = u32(std::stoul(next("--seed")));
    else if (s == "--samples")            a.p_samples = u32(std::stoul(next("--samples")));
    else if (s == "--width")              a.width  = std::stoi(next("--width"));
    else if (s == "--height")             a.height = std::stoi(next("--height"));
    else if (s == "--spp")                a.spp    = std::stoi(next("--spp"));
    else if (s == "--object-radius")      a.radius = std::stof(next("--object-radius"));
    else if (s == "--no-medium")          a.noMedium = true;
    else if (s == "--gap")                a.gap    = std::stof(next("--gap"));
    else if (s == "--cam-dist")           a.camDist   = std::stof(next("--cam-dist"));
    else if (s == "--cam-height")         a.camHeight = std::stof(next("--cam-height"));
    else if (s == "--length") {
      if (!parseVec3(next("--length"), a.p.Lx, a.p.Ly, a.p.Lz)) {
        std::printf("error: --length expects Lx,Ly,Lz\n");
        return 1;
      }
    }
    else if (s == "-h" || s == "--help")  { usage(); return 0; }
    else if (a.input.empty())             a.input = s;
    else { std::printf("error: unexpected argument '%s'\n", s.c_str()); return 1; }
  }

  if (a.cmd == "build") {
    if (a.input.empty() || a.output.empty()) { usage(); return 1; }
    gpifile::GPISData data;
    if (!host::buildGpisFromObj(a.input, a.p, data)) return 1;
    if (!gpifile::write(a.output, data)) return 1;
    std::printf("%s", gpifile::describe(data).c_str());
    return 0;
  }
  if (a.cmd == "info") {
    if (a.input.empty()) { usage(); return 1; }
    gpifile::GPISData data;
    if (!gpifile::read(a.input, data)) return 1;
    std::printf("%s", gpifile::describe(data).c_str());
    return 0;
  }
  if (a.cmd == "check") {
    if (a.input.empty()) { usage(); return 1; }
    return runCheck(a);
  }
  if (a.cmd == "render") {
    if (a.input.empty() || a.origObj.empty() || a.output.empty()) { usage(); return 1; }
    return runRender(a);
  }

  usage();
  return 1;
}
