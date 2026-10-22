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
                core::kernelColor(p, bounds.min, bounds.max);
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
    m.boundCenter  = {0.0f, 0.0f, -500.0f};
    m.boundRadius  = 170.0f;                 // wrapper, ≥ mean + a few σ
    m.sceneScale   = 150.0f;                 // ≈ object radius
    m.sigma        = 5.0f;
    m.lengthScale  = {1.0f, 1.0f, 1.0f};
    m.kernelRadius = 3.0f;                   // cell size
    m.impulseDensity = 3;                    // points_per_cell
    m.gpMode       = core::GPMode::RenewalPlus;
    cfg.hasMedium  = true;

    // ——— primitives ———
    // Wrapping sphere: transparent (Null), contains the GP medium (medium id 0).
    cfg.host.addSphere({0.0f, 0.0f, -500.0f}, 170.0f,
                       u16(core::BsdfType::Null), 0, core::NO_LIGHT);

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

  SceneConfig buildMeshGpScene(const std::string& objPath, i32 w, i32 h,
                               f32 dist, f32 targetRadius, core::GPMode mode,
                               f32 sigma, f32 kernelRadius, bool useVoxel, u32 voxelRes,
                               bool nonStationary) {
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

    const u32 nt = u32(obj.faces.size() / 3);
    cfg.host.addMeshMean(worldVerts.data(), u32(worldVerts.size()), obj.faces.data(), nt);
    mplog("buildMeshGpScene: {} verts {} tris (scale {:.3f})", worldVerts.size(), nt, scale);

    // ——— GP medium: mesh/voxel mean + sparse SE convolution noise ———
    core::Medium& m = cfg.medium;
    m.type         = core::MediumType::GP;
    m.meanType     = core::MeanType::Mesh;
    m.boundCenter  = pos;
    m.boundRadius  = targetRadius * 1.4f;
    m.sceneScale   = targetRadius;
    m.sigma        = sigma;
    m.lengthScale  = {1.0f, 1.0f, 1.0f};
    m.kernelRadius = kernelRadius;
    m.impulseDensity = 3;
    m.gpMode       = mode;
    cfg.hasMedium  = true;

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

      mps::host::VoxelGridHost vg = mps::host::buildSparseSDFGrid(mm, worldBox, voxelRes, 4.0f);
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

    // Bake the dense RGB color grid (GPIS texturing approach 1), cheap at res 64.
    {
      core::AABB worldBox3 = core::AABB::empty();
      for (const vec3f& v : worldVerts) worldBox3.extend(v);
      buildColorGrid(cfg, worldBox3, 64);
    }

    // ——— wrapper sphere ———
    // Removed: the medium bound (boundCenter/boundRadius) is now tested directly
    // in trace.cpp via sphereEnterT(), so no Null wrapper primitive is needed (and
    // it can never be accidentally shaded as an opaque sphere).

    // ——— area lights (sampled by NEE) ———
    cfg.host.addLight({-300.0f, 350.0f, -350.0f}, 70.0f, {5.0f, 4.0f, 3.5f});   // key
    cfg.host.addLight({ 300.0f, 100.0f, -150.0f}, 45.0f, {2.0f, 1.6f, 1.4f});   // fill

    // ——— floor + back wall (raised floor + taller wall so they fill the frame) ———
    addFloorWalls(cfg, -150.0f, -850.0f, 550.0f);

    // ——— camera: straight ahead ———
    cfg.camera = core::makeCamera({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f},
                                  45.0f, w, h);

    return cfg;
  }

  SceneConfig buildSpotGpScene(i32 w, i32 h) {
    return buildMeshGpScene("/home/dudujerry/models/spot/spot_triangulated_good.obj",
                            w, h, 500.0f, 150.0f, core::GPMode::SingleRealization, 1.0f, 1.0f);
  }

  SceneConfig buildBunnyGpScene(i32 w, i32 h) {
    return buildMeshGpScene("/home/dudujerry/models/bunny/bunny.obj",
                            w, h, 500.0f, 150.0f, core::GPMode::SingleRealization, 1.0f, 1.0f);
  }

  SceneConfig buildDragonGpScene(i32 w, i32 h) {
    return buildMeshGpScene("/home/dudujerry/models/dragon.obj",
                            w, h, 500.0f, 150.0f, core::GPMode::SingleRealization, 1.0f, 1.0f);
  }

}
