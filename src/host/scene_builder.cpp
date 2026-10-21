#include "host/scene_builder.h"

#include <vector>

#include "core/bsdf/bsdf.h"
#include "core/primitive/primitive.h"
#include "core/utils/log.h"
#include "host/obj_loader.h"
#include "host/voxel_builder.h"

namespace mps::host {

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
    cfg.host.addLight({-250.0f, 350.0f, -350.0f}, 70.0f, {40.0f, 36.0f, 30.0f});  // key
    cfg.host.addLight({ 300.0f,  80.0f, -200.0f}, 45.0f, {15.0f, 12.0f, 10.0f});  // fill

    // ——— camera ———
    cfg.camera = core::makeCamera({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f},
                                  45.0f, w, h);

    return cfg;
  }

  SceneConfig buildMeshGpScene(const std::string& objPath, i32 w, i32 h,
                               f32 dist, f32 targetRadius, core::GPMode mode,
                               f32 sigma, f32 kernelRadius, bool useVoxel, u32 voxelRes) {
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

    // ——— wrapper sphere (transparent, holds the medium) ———
    cfg.host.addSphere(pos, m.boundRadius, u16(core::BsdfType::Null), 0, core::NO_LIGHT);

    // ——— area lights (sampled by NEE) ———
    cfg.host.addLight({-300.0f, 350.0f, -350.0f}, 70.0f, {40.0f, 36.0f, 30.0f});  // key
    cfg.host.addLight({ 300.0f, 100.0f, -150.0f}, 45.0f, {15.0f, 12.0f, 10.0f});  // fill

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
