#include "host/scene_builder.h"

#include <vector>

#include "core/bsdf/bsdf.h"
#include "core/primitive/primitive.h"
#include "core/utils/log.h"
#include "host/obj_loader.h"

namespace mps::host {

  SceneConfig buildDefaultScene() {
    SceneConfig cfg;

    cfg.settings.maxBounce = 10;
    cfg.settings.eps      = 1e-3f;
    cfg.settings.nearEps  = 1e-2f;
    cfg.spp = 1;

    // ——— GP medium (sphere mean, sparse SE convolution noise) ———
    core::Medium& m = cfg.medium;
    m.type         = core::MediumType::GP;
    m.meanType     = core::MeanType::Sphere;
    m.meanCenter   = {0.0f, 0.0f, -500.0f};
    m.meanRadius   = 150.0f;
    m.boundCenter  = {0.0f, 0.0f, -500.0f};
    m.boundRadius  = 170.0f;                 // wrapper, ≥ mean + a few σ
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

    // ——— camera ———
    const i32 w = 160, h = 120;
    cfg.camera = core::makeCamera({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f},
                                  45.0f, w, h);

    return cfg;
  }

  SceneConfig buildSpotGpScene() {
    SceneConfig cfg;

    cfg.settings.maxBounce = 10;
    cfg.settings.eps      = 1e-3f;
    cfg.settings.nearEps  = 1e-2f;
    cfg.spp = 1;

    // ——— load + transform the spot mesh (world space = obj * scale + pos) ———
    const char* path = "/home/dudujerry/models/spot/spot_triangulated_good.obj";
    const f32 scale = 110.0f;
    const vec3f pos = {0.0f, 0.0f, -500.0f};

    ObjMesh obj;
    if (!loadObj(path, obj)) {
      mplog("buildSpotGpScene: failed to load {}", path);
      return cfg;
    }

    std::vector<vec3f> worldVerts(obj.vertices.size());
    for (size_t i = 0; i < obj.vertices.size(); ++i)
      worldVerts[i] = obj.vertices[i] * scale + pos;

    const u32 nt = u32(obj.faces.size() / 3);
    cfg.host.addMeshMean(worldVerts.data(), u32(worldVerts.size()), obj.faces.data(), nt);
    mplog("buildSpotGpScene: spot mesh {} verts {} tris", worldVerts.size(), nt);

    // ——— GP medium: mesh mean + sparse SE convolution noise ———
    core::Medium& m = cfg.medium;
    m.type         = core::MediumType::GP;
    m.meanType     = core::MeanType::Mesh;
    m.meanCenter   = {0.0f, 0.0f, 0.0f};     // unused for mesh mean
    m.meanRadius   = 1.0f;                   // unused for mesh mean
    m.boundCenter  = {0.0f, 0.0f, -500.0f};
    m.boundRadius  = 200.0f;                 // wrapper sphere around the scaled spot
    m.sigma        = 1.0f;
    m.lengthScale  = {1.0f, 1.0f, 1.0f};
    m.kernelRadius = 1.0f;                   // cell size (matches original spot scene)
    m.impulseDensity = 3;                    // points_per_cell
    m.gpMode       = core::GPMode::SingleRealization;
    cfg.hasMedium  = true;

    // ——— wrapper sphere (transparent, holds the medium) ———
    cfg.host.addSphere({0.0f, 0.0f, -500.0f}, 200.0f,
                       u16(core::BsdfType::Null), 0, core::NO_LIGHT);

    // ——— camera: straight ahead, object centered on -z ———
    const i32 w = 256, h = 256;
    cfg.camera = core::makeCamera({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f},
                                  45.0f, w, h);

    return cfg;
  }

}
