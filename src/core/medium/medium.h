#pragma once

#include "core/basic/core.h"
#include "core/geometry/ray.h"
#include "core/math/math_util.h"
#include "core/gp/meanfunction.h"

namespace mps::core {

  // Renewal+ conditioning state carried along a single path.
  // ψ_cond(x) = ψ(x) + α·h(C,x) + β·∇h(C,x)
  struct GPConditioningState {
    bool  active = false;
    vec3f C{0.0f, 0.0f, 0.0f};   // conditioning (surface) point
    f32   alpha = 0.0f;           // scalar  correction (pin f(C)=0)
    vec3f beta{0.0f, 0.0f, 0.0f}; // gradient correction (match ∇f(C)=n)
  };

  struct MediumSample {
    bool  exited = true;
    f32   t = 0.0f;
    vec3f p{0.0f, 0.0f, 0.0f};
    vec3f normal{0.0f, 0.0f, 0.0f};
    // Unconditioned gradient decomposition at the hit (for §6 NDF NEE):
    // gradMean = ∇μ, gradNoise = ∇ψ (sampled); g_z = dot(gradNoise, ray.dir).
    vec3f gradMean{0.0f, 0.0f, 0.0f};
    vec3f gradNoise{0.0f, 0.0f, 0.0f};
  };

  enum class MediumType : u16 { SDF = 0, GP = 1 };
  enum class MeanType   : u16 { Sphere = 0, Mesh = 1, Voxel = 2 };
  enum class GPMode     : u16 { SingleRealization = 0, RenewalPlus = 1 };

  // POD medium parameters (device-friendly). The GP medium wraps a mean
  // (sphere, this stage) + an SE kernel + sparse convolution noise.
  struct Medium {
    MediumType type = MediumType::GP;
    MeanType   meanType = MeanType::Sphere;

    vec3f meanCenter{0.0f, 0.0f, 0.0f};
    f32   meanRadius = 1.0f;

    // Mesh mean (used when meanType == MeanType::Mesh). Host/device pointers are
    // bound by the backend after scene upload.
    MeshMean meshMean;

    // Sparse narrow-band SDF mean (used when meanType == MeanType::Voxel).
    // Built on the host from the mesh, uploaded + re-bound like meshMean.
    SparseSDFGrid voxelMean;

    // Bounding (wrapper) sphere: the medium only lives inside it, so ray
    // marching is clamped to the far intersection with this sphere.
    vec3f boundCenter{0.0f, 0.0f, 0.0f};
    f32   boundRadius = 1.0f;

    // Characteristic scene length (≈ object radius). Scale-aware marching
    // constants in constant.h are multiplied by this at the use site.
    f32 sceneScale = 1.0f;

    f32   sigma = 1.0f;
    vec3f lengthScale{1.0f, 1.0f, 1.0f};
    f32   kernelRadius = 1.0f;
    u32   impulseDensity = 3;   // points_per_cell

    GPMode gpMode = GPMode::SingleRealization;

    // BSDF of the GP implicit surface (BsdfType: 1=Lambertian, 2=Specular).
    // A specular GPIS reflects like a mirror about the noisy GP normal — the
    // "perfect specular microsurface" of the paper (Fig 12/15).
    u16 surfaceBsdf = 1;   // = BsdfType::Lambertian
  };

  // Far intersection of a ray with a sphere; ray.max if it misses.
  // Used to clamp the march to the wrapper (bounding) sphere.
  MPHD MPINL f32 sphereExitT(const Ray& ray, const vec3f& C, f32 R) {
    vec3f V = ray.o - C;
    f32 k = mps::dot(ray.dir, V);
    f32 disc = k * k - (mps::dot(V, V) - R * R);
    if (disc < 0.0f) return ray.max;
    f32 s = mps::sqrtf(disc);
    f32 t2 = -k + s;   // far root
    return t2 > ray.min ? t2 : ray.max;
  }

  // Ray-marching entry points (dispatch on m.type).
  // sampleDistance: returns true and fills `out` when the ray crosses the
  // implicit surface inside [ray.min, ray.max]; false if it exits.
  MPHD bool    sampleDistance(const Ray& ray, const Medium& m, const GPConditioningState& c, u32 seed, MediumSample& out);
  MPHD RGB_F32 transmittance (const Ray& ray, const Medium& m, const GPConditioningState& c, u32 seed);
  MPHD GPConditioningState sampleCondition(const MediumSample& s, const Medium& m, u32 seed, const vec3f& rayDir);

}
