#include "host/scene_builder.h"

#include "core/bsdf/bsdf.h"
#include "core/primitive/primitive.h"

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

}
