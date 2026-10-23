#include "host/scene_builder.h"

#include <vector>

#include "core/bsdf/bsdf.h"
#include "core/primitive/primitive.h"
#include "core/utils/log.h"
#include "host/obj_loader.h"
#include "host/voxel_builder.h"

namespace mps::host {

  namespace {

    // Add a quad (2 triangles) with the given BSDF as a hard surface.
    void addQuad(core::SceneHost& h, const vec3f& a, const vec3f& b, const vec3f& c, const vec3f& d,
                 u16 bsdfId) {
      const vec3f verts[4] = {a, b, c, d};
      const u32 tris[6] = {0, 1, 2, 0, 2, 3};
      h.addMesh(verts, 4, tris, 2, bsdfId, core::NO_MEDIUM, core::NO_LIGHT);
    }

    // Floor + back wall, placed below/behind the GP wrapper sphere so they stay
    // outside the GP medium.
    void addFloorWalls(SceneConfig& cfg, f32 floorY, f32 backZ, f32 extent) {
      addQuad(cfg.host, {-extent, floorY, -200.0f}, {extent, floorY, -200.0f},
              {extent, floorY, backZ}, {-extent, floorY, backZ},
              u16(core::BsdfType::Lambertian));
      addQuad(cfg.host, {-extent, floorY, backZ}, {extent, floorY, backZ},
              {extent, floorY + extent, backZ}, {-extent, floorY + extent, backZ},
              u16(core::BsdfType::Lambertian));
    }

    // Procedural non-stationary kernel field for Eq. 25 testing: length scale L
    // ramps linearly with height (fine at the bottom, coarse at the top); σ is
    // constant. Written into cfg.host.kernel* and cfg.medium.kernelField.
    void buildTestKernelField(SceneConfig& cfg, const core::AABB& bounds, u32 res,
                              f32 baseSigma, f32 baseLength) {
      const vec3f diag = bounds.diagonal();
      const f32 maxExtent = mps::maxf(diag.x, mps::maxf(diag.y, diag.z));
      const f32 cellSize = (res > 0) ? maxExtent / f32(res) : 1.0f;
      const f32 minY = bounds.min.y, maxY = bounds.max.y;

      cfg.host.kernelRes = res;
      cfg.host.kernelSigma.assign(size_t(res) * res * res, 0.0f);
      cfg.host.kernelLength.assign(size_t(res) * res * res, 0.0f);

      for (u32 k = 0; k < res; ++k)
        for (u32 j = 0; j < res; ++j)
          for (u32 i = 0; i < res; ++i) {
            const f32 y = minY + f32(j) * cellSize;
            const f32 t = (maxY > minY) ? (y - minY) / (maxY - minY) : 0.5f;
            const f32 L = baseLength * (0.5f + 1.5f * t);
            const size_t idx = size_t(i) + res * (size_t(j) + res * size_t(k));
            cfg.host.kernelSigma[idx]  = baseSigma;
            cfg.host.kernelLength[idx] = L;
          }

      cfg.medium.kernelField.origin   = bounds.min;
      cfg.medium.kernelField.cellSize = cellSize;
      cfg.medium.kernelField.res      = res;
      mplog("buildTestKernelField: res={} L in [{:.2f}, {:.2f}]", res,
            baseLength * 0.5f, baseLength * 2.0f);
    }

    // Non-stationary SIGMA field: σ ramps monotonically along one world axis
    // (0=x, 1=y, 2=z) from sigmaMin (at the MESH bbox min along the axis) to
    // sigmaMax (at the MESH bbox max). The ramp is normalized by the mesh's own
    // extent along that axis — NOT the AABB — so the bottom of the model gets
    // sigmaMin and the top gets sigmaMax.
    void buildSigmaRampField(SceneConfig& cfg, const core::AABB& bounds, u32 res,
                             f32 sigmaMin, f32 sigmaMax, f32 baseLength, i32 axis,
                             bool sigmaClamp) {
      const vec3f diag = bounds.diagonal();
      const f32 maxExtent = mps::maxf(diag.x, mps::maxf(diag.y, diag.z));
      const f32 cellSize = (res > 0) ? maxExtent / f32(res) : 1.0f;

      cfg.host.kernelRes = res;
      cfg.host.kernelSigma.assign(size_t(res) * res * res, 0.0f);
      cfg.host.kernelLength.assign(size_t(res) * res * res, 0.0f);

      const vec3f mn = bounds.min, mx = bounds.max;
      const f32 span = mps::maxf(1e-6f, (axis == 0 ? mx.x - mn.x :
                                         axis == 1 ? mx.y - mn.y : mx.z - mn.z));
      const f32 mnCoord = (axis == 0 ? mn.x : axis == 1 ? mn.y : mn.z);
      const char axisName = "xyz"[(axis == 0 ? 0 : axis == 1 ? 1 : 2)];
      for (u32 k = 0; k < res; ++k)
        for (u32 j = 0; j < res; ++j)
          for (u32 i = 0; i < res; ++i) {
            const vec3f p = mn + vec3f{f32(i), f32(j), f32(k)} * cellSize;
            const f32 coord = (axis == 0 ? p.x : axis == 1 ? p.y : p.z);
            const f32 t = mps::clampf((coord - mnCoord) / span, 0.0f, 1.0f);
            // clamp mode: hard step at the model midpoint (t=0.5), else linear ramp.
            const f32 sig = sigmaClamp ? (t < 0.5f ? sigmaMin : sigmaMax)
                                       : (sigmaMin + (sigmaMax - sigmaMin) * t);
            const size_t idx = size_t(i) + res * (size_t(j) + res * size_t(k));
            cfg.host.kernelSigma[idx]  = sig;
            cfg.host.kernelLength[idx] = baseLength;
          }

      cfg.medium.kernelField.origin   = mn;
      cfg.medium.kernelField.cellSize = cellSize;
      cfg.medium.kernelField.res      = res;
      mplog("buildSigmaRampField: axis={} {} sigma in [{:.2f}, {:.2f}] over mesh bbox",
            axisName, sigmaClamp ? "CLAMP(step@0.5)" : "ramp", sigmaMin, sigmaMax);
    }

    // Bake the analytic 3D color field into a dense RGB voxel grid (approach 1).
    void buildColorGrid(SceneConfig& cfg, const core::AABB& bounds, u32 res) {      const vec3f diag = bounds.diagonal();
      const f32 maxExtent = mps::maxf(diag.x, mps::maxf(diag.y, diag.z));
      const f32 cellSize = (res > 0) ? maxExtent / f32(res) : 1.0f;

      cfg.host.colorRes = res;
      cfg.host.colorGrid.assign(size_t(res) * res * res, RGB_F32{0, 0, 0});
      for (u32 k = 0; k < res; ++k)
        for (u32 j = 0; j < res; ++j)
          for (u32 i = 0; i < res; ++i) {
            const vec3f p = bounds.min + vec3f{f32(i), f32(j), f32(k)} * cellSize;
            cfg.host.colorGrid[size_t(i) + res * (size_t(j) + res * size_t(k))] =
                core::fieldColor(p, bounds.min, bounds.max);
          }

      cfg.medium.colorGrid.origin   = bounds.min;
      cfg.medium.colorGrid.cellSize = cellSize;
      cfg.medium.colorGrid.res      = res;
      mplog("buildColorGrid: res={} baked", res);
    }

  } // namespace

  SceneConfig buildDefaultScene(i32 w, i32 h) {
    SceneConfig cfg;

    cfg.settings.maxBounce = 10;
    cfg.settings.eps      = core::EPS_SURFACE;
    cfg.settings.nearEps  = core::EPS_NEAR;
    cfg.spp = 1;

    // ——— GP medium (sphere mean, sparse SE convolution noise) ———
    core::Medium& m = cfg.medium;
    m.type         = core::MediumType::GP;
    m.meanType     = core::MeanType::Sphere;
    m.meanCenter   = {0.0f, 0.0f, -500.0f};
    m.meanRadius   = 150.0f;
    m.sceneScale   = 150.0f;                 // ≈ object radius
    m.sigma        = 5.0f;
    m.lengthScale  = {1.0f, 1.0f, 1.0f};
    m.kernelRadius = 3.0f;                   // cell size
    m.impulseDensity = 3;                    // points_per_cell
    m.gpMode       = core::GPMode::RenewalPlus;
    cfg.hasMedium  = true;

    // AABB medium bound: mean sphere + a few σ of GP noise; bottom face on the
    // floor so the medium never crosses it.
    {
      const f32 half = m.meanRadius + 3.0f * m.sigma;
      m.boundMin = m.meanCenter - vec3f{half, half, half};
      m.boundMax = m.meanCenter + vec3f{half, half, half};
      m.boundMin.y = -300.0f;   // floor
    }

    // ——— primitives ———
    // (the medium bound is tested directly in trace.cpp via the AABB; no
    // wrapper primitive is needed or added)

    // Solid Lambertian sphere for contrast (no medium).
    cfg.host.addSphere({120.0f, 60.0f, -280.0f}, 60.0f,
                       u16(core::BsdfType::Lambertian), core::NO_MEDIUM, core::NO_LIGHT);

    // Specular sphere: tests the delta-BSDF mirror path.
    cfg.host.addSphere({-60.0f, 60.0f, -340.0f}, 75.0f,
                       u16(core::BsdfType::Specular), core::NO_MEDIUM, core::NO_LIGHT);

    // ——— area lights (sampled by NEE) ———
    cfg.host.addLight({-250.0f, 350.0f, -350.0f}, 70.0f, {26.0f, 23.0f, 19.0f});  // key
    cfg.host.addLight({ 300.0f,  80.0f, -200.0f}, 45.0f, {10.0f, 8.0f, 7.0f});   // fill

    // ——— floor + back wall ———
    addFloorWalls(cfg, -300.0f, -850.0f, 500.0f);

    // ——— camera ———
    cfg.camera = core::makeCamera({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f},
                                  45.0f, w, h);

    return cfg;
  }

  SceneConfig buildGpSphereScene(i32 w, i32 h, f32 sigma) {
    SceneConfig cfg;

    cfg.settings.maxBounce = 10;
    cfg.settings.eps      = core::EPS_SURFACE;
    cfg.settings.nearEps  = core::EPS_NEAR;
    cfg.spp = 1;

    // ——— GP medium: sphere mean + sparse SE convolution noise ———
    core::Medium& m = cfg.medium;
    m.type           = core::MediumType::GP;
    m.meanType       = core::MeanType::Sphere;
    m.meanCenter     = {0.0f, 0.0f, -500.0f};
    m.meanRadius     = 150.0f;
    m.sceneScale     = 150.0f;
    m.sigma          = sigma;
    m.lengthScale    = {1.0f, 1.0f, 1.0f};
    m.kernelRadius   = 3.0f;
    m.impulseDensity = 3;
    m.gpMode         = core::GPMode::RenewalPlus;
    cfg.hasMedium    = true;

    {
      const f32 half = m.meanRadius + 3.0f * m.sigma;
      m.boundMin = m.meanCenter - vec3f{half, half, half};
      m.boundMax = m.meanCenter + vec3f{half, half, half};
      m.boundMin.y = -300.0f;
    }

    // ——— area lights (sampled by NEE) ———
    cfg.host.addLight({-250.0f, 350.0f, -350.0f}, 70.0f, {26.0f, 23.0f, 19.0f});
    cfg.host.addLight({ 300.0f,  80.0f, -200.0f}, 45.0f, {10.0f, 8.0f, 7.0f});

    // ——— floor + back wall ———
    addFloorWalls(cfg, -300.0f, -850.0f, 500.0f);

    cfg.camera = core::makeCamera({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f},
                                  45.0f, w, h);
    return cfg;
  }

  SceneConfig buildMeshGpScene(const std::string& objPath, i32 w, i32 h,
                               f32 dist, f32 targetRadius, core::GPMode mode,
                               f32 sigma, f32 kernelRadius, bool useVoxel, u32 voxelRes,
                               bool nonStationary, i32 sigmaFieldAxis, f32 sigmaMin, f32 sigmaMax,
                               bool sigmaClamp, vec3f marginScale) {
    SceneConfig cfg;

    cfg.settings.maxBounce = 10;
    cfg.settings.eps      = core::EPS_SURFACE;
    cfg.settings.nearEps  = core::EPS_NEAR;
    cfg.spp = 1;

    ObjMesh obj;
    if (!loadObj(objPath, obj)) {
      mplog("buildMeshGpScene: failed to load {}", objPath);
      return cfg;
    }

    // Auto-fit: center the mesh and scale it into a sphere of `targetRadius`.
    core::AABB box = core::AABB::empty();
    for (const vec3f& v : obj.vertices) box.extend(v);
    const vec3f center = box.centroid();
    const vec3f diag   = box.diagonal();
    const f32 diagLen  = mps::norm(diag);
    const f32 scale    = (diagLen > 0.0f) ? (2.0f * targetRadius / diagLen) : 1.0f;
    const vec3f pos    = {0.0f, 0.0f, -dist};

    std::vector<vec3f> worldVerts(obj.vertices.size());
    for (size_t i = 0; i < obj.vertices.size(); ++i)
      worldVerts[i] = (obj.vertices[i] - center) * scale + pos;

    // Sit the mesh on the floor: translate it down so its lowest vertex is at
    // the floor plane (the auto-fit centers it, which leaves an oblong mesh
    // floating above the floor).
    const f32 floorY = -150.0f;
    {
      core::AABB wb = core::AABB::empty();
      for (const vec3f& v : worldVerts) wb.extend(v);
      const f32 drop = wb.min.y - floorY;
      for (vec3f& v : worldVerts) v.y -= drop;
    }

    const u32 nt = u32(obj.faces.size() / 3);
    cfg.host.addMeshMean(worldVerts.data(), u32(worldVerts.size()), obj.faces.data(), nt);
    mplog("buildMeshGpScene: {} verts {} tris (scale {:.3f})", worldVerts.size(), nt, scale);

    // ——— GP medium: mesh/voxel mean + sparse SE convolution noise ———
    core::Medium& m = cfg.medium;
    m.type         = core::MediumType::GP;
    m.meanType     = core::MeanType::Mesh;
    m.sceneScale   = targetRadius;
    m.sigma        = sigma;
    m.lengthScale  = {1.0f, 1.0f, 1.0f};
    m.kernelRadius = kernelRadius;
    m.impulseDensity = 3;
    m.gpMode       = mode;
    cfg.hasMedium  = true;

    // With a sigma ramp the AABB margin and voxel narrow band must be sized for
    // the field's PEAK (sigmaMax), not the uniform base sigma.
    const f32 maxSigma = (sigmaFieldAxis >= 0) ? sigmaMax : sigma;

    // AABB medium bound: mesh bbox + a few σ of GP noise, with the bottom face
    // exactly on the floor (so the medium never crosses the floor plane — no
    // circular shadow gap, unlike a bounding sphere).
    {
      core::AABB wb = core::AABB::empty();
      for (const vec3f& v : worldVerts) wb.extend(v);
      // Per-axis AABB margin (multiples of σ), default {3,5,3}: the top gets
      // extra headroom because at large σ the GP surface puffs upward (ears
      // especially) past 3σ and any zero-set outside the AABB is clipped off.
      m.boundMin = wb.min - vec3f{marginScale.x * maxSigma, marginScale.y * maxSigma, marginScale.z * maxSigma};
      m.boundMax = wb.max + vec3f{marginScale.x * maxSigma, marginScale.y * maxSigma, marginScale.z * maxSigma};
      // Keep the AABB's bottom a hair ABOVE the floor so the floor plane is
      // OUTSIDE the medium bound: a ray marching downward then exits the box
      // first and still hits the floor (instead of skipping past it).
      m.boundMin.y = floorY + 0.5f;
    }

    if (useVoxel) {
      // World-space AABB of the auto-fitted mesh.
      core::AABB worldBox = core::AABB::empty();
      for (const vec3f& v : worldVerts) worldBox.extend(v);

      // Build a temporary MeshMean over the host arrays (before toDevice).
      core::MeshMean mm;
      mm.vertices       = cfg.host.vertices.data();
      mm.faces          = cfg.host.faces.data();
      mm.bvhNodes       = cfg.host.bvhNodes.data();
      mm.bvhTriIndices  = cfg.host.bvhTriIndices.data();
      mm.vertexNormals  = cfg.host.vertexNormals.data();
      mm.bvhRoot        = cfg.host.meshMeans[0].bvhRoot;
      mm.bvhNodeOffset  = cfg.host.meshMeans[0].bvhNodeOffset;
      mm.bvhTriOffset   = cfg.host.meshMeans[0].bvhTriOffset;

      // Narrow band sized to the GP noise amplitude: the ±band fallback of the
      // sparse grid must exceed ~3σ so the noise does not create spurious zero
      // surfaces far from the mesh (otherwise large σ fills the whole AABB).
      const vec3f wdiag = worldBox.diagonal();
      const f32 maxExt = mps::maxf(wdiag.x, mps::maxf(wdiag.y, wdiag.z));
      const f32 cellSz = (voxelRes > 0) ? maxExt / f32(voxelRes) : 1.0f;
      const f32 bandCells = (4.0f * maxSigma) / cellSz + 2.0f;   // 4σ (world) + margin, in cells

      mps::host::VoxelGridHost vg = mps::host::buildSparseSDFGrid(mm, worldBox, voxelRes, bandCells);
      cfg.host.voxelRes       = vg.res;
      cfg.host.voxelSignGrid  = std::move(vg.signGrid);
      cfg.host.voxelBandIndex = std::move(vg.bandIndex);
      cfg.host.voxelBandValue = std::move(vg.bandValue);

      m.voxelMean.origin   = vg.origin;
      m.voxelMean.cellSize = vg.cellSize;
      m.voxelMean.band     = vg.band;
      m.voxelMean.res      = vg.res;
      m.meanType           = core::MeanType::Voxel;
    }

    if (nonStationary) {
      // Non-stationary kernel field over the same world bounds.
      core::AABB worldBox2 = core::AABB::empty();
      for (const vec3f& v : worldVerts) worldBox2.extend(v);
      buildTestKernelField(cfg, worldBox2, 64, sigma, kernelRadius);
    }

    if (sigmaFieldAxis >= 0) {
      // Spatially-varying SIGMA ramp along one axis over the MESH bbox (bottom ->
      // top), sigmaMin -> sigmaMax. Length scale stays constant.
      core::AABB worldBox4 = core::AABB::empty();
      for (const vec3f& v : worldVerts) worldBox4.extend(v);
      buildSigmaRampField(cfg, worldBox4, 64, sigmaMin, sigmaMax, kernelRadius, sigmaFieldAxis, sigmaClamp);
    }

    // Bake the dense RGB color grid (GPIS texturing approach 1), cheap at res 64.
    {
      core::AABB worldBox3 = core::AABB::empty();
      for (const vec3f& v : worldVerts) worldBox3.extend(v);
      buildColorGrid(cfg, worldBox3, 64);
    }

    // ——— medium bound ———
    // The medium bound AABB is tested directly in trace.cpp (aabbIntersect), so
    // no wrapper primitive is needed (and none can be accidentally shaded).

    // ——— area lights (sampled by NEE) ———
    cfg.host.addLight({ -300.0f, 350.0f, -350.0f}, 70.0f, {5.0f, 4.0f, 3.5f});   // key
    cfg.host.addLight({ 300.0f, 100.0f, -150.0f}, 45.0f, {2.0f, 1.6f, 1.4f});   // fill

    // ——— floor + back wall (raised floor + taller wall so they fill the frame) ———
    addFloorWalls(cfg, -150.0f, -850.0f, 550.0f);

    // ——— camera: straight ahead ———
    cfg.camera = core::makeCamera({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f},
                                  45.0f, w, h);

    return cfg;
  }

  SceneConfig buildSpotGpScene(i32 w, i32 h, f32 sigma, i32 sigmaFieldAxis, f32 sigmaMin, f32 sigmaMax, bool sigmaClamp, vec3f marginScale) {
    return buildMeshGpScene("/home/dudujerry/models/spot/spot_triangulated_good.obj",
                            w, h, 500.0f, 150.0f, core::GPMode::SingleRealization, sigma, 1.0f,
                            true, 256, false, sigmaFieldAxis, sigmaMin, sigmaMax, sigmaClamp, marginScale);
  }

  SceneConfig buildBunnyGpScene(i32 w, i32 h, f32 sigma, i32 sigmaFieldAxis, f32 sigmaMin, f32 sigmaMax, bool sigmaClamp, vec3f marginScale) {
    return buildMeshGpScene("/home/dudujerry/models/bunny/bunny.obj",
                            w, h, 500.0f, 150.0f, core::GPMode::SingleRealization, sigma, 1.0f,
                            true, 256, false, sigmaFieldAxis, sigmaMin, sigmaMax, sigmaClamp, marginScale);
  }

  SceneConfig buildDragonGpScene(i32 w, i32 h, f32 sigma, i32 sigmaFieldAxis, f32 sigmaMin, f32 sigmaMax, bool sigmaClamp, vec3f marginScale) {
    return buildMeshGpScene("/home/dudujerry/models/dragon.obj",
                            w, h, 500.0f, 150.0f, core::GPMode::SingleRealization, sigma, 1.0f,
                            true, 256, false, sigmaFieldAxis, sigmaMin, sigmaMax, sigmaClamp, marginScale);
  }

}
