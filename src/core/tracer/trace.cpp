#include "core/tracer/trace.h"

#include "core/bsdf/bsdf.h"
#include "core/math/random.h"

namespace mps::core {

  MPHD RGB_F32 trace(const Ray& ray0, const RenderContext& ctx, u32 seed) {
    Ray ray = ray0;
    RGB_F32 emission{0.0f, 0.0f, 0.0f};
    RGB_F32 throughput{1.0f, 1.0f, 1.0f};
    GPConditioningState cond;

    // Simple direct illumination: a fixed sun + ambient (no NEE in this milestone).
    const vec3f sunDir = mps::normalize(vec3f{0.3f, 1.0f, 0.2f});
    const RGB_F32 sun{1.4f, 1.3f, 1.1f};
    const RGB_F32 ambient{0.06f, 0.06f, 0.08f};

    Random rng(seed);
    bool inMedium = ctx.hasMedium;

    for (i32 bounce = 0; bounce < ctx.settings.maxBounce; ++bounce) {

      // ——— ray marching branch: inside the wrapped medium ———
      if (inMedium) {
        MediumSample ms;
        if (sampleDistance(ray, ctx.medium, cond, seed, ms)) {
          // Hit the implicit surface; shade with the gradient as the normal.
          RGB_F32 albedo = lambertianAlbedo();
          f32 ndl = mps::fabsf(mps::dot(ms.normal, sunDir));
          emission += throughput * albedo * (ambient + sun * ndl);

          // Scatter (Lambertian), then pin the surface for the next bounce.
          BsdfSample bs = bsdfSample(BsdfType::Lambertian, -ray.dir, ms.normal, {0.0f, 0.0f},
                                     rng.next1D(), rng.next1D());
          throughput = throughput * bs.weight;

          cond = sampleCondition(ms, ctx.medium, seed);

          ray.o = ms.p + ms.normal * ctx.settings.eps;
          ray.dir = bs.wi;
          ray.min = ctx.settings.nearEps;
          ray.max = 1e30f;
          continue;
        } else {
          inMedium = false;   // exited the medium; fall through to hard surfaces
        }
      }

      // ——— ray tracing branch: hard surfaces ———
      HitRecord hr = sceneIntersect(ray, ctx.scene);
      if (!hr.hit) {
        emission += throughput * background(ray.dir);
        break;
      }

      BsdfType bt = BsdfType(hr.bsdfId);
      BsdfSample bs = bsdfSample(bt, -ray.dir, hr.ng, hr.uv, rng.next1D(), rng.next1D());

      // Entering a wrapped medium (transparent boundary, wi points against the normal).
      if (mps::dot(bs.wi, hr.ng) < 0.0f && hr.mediumId != NO_MEDIUM) {
        inMedium = true;
        ray.o = hr.p - hr.ng * ctx.settings.eps;  // step just inside
        ray.dir = bs.wi;
        ray.min = ctx.settings.nearEps;
        // Clamp the march to the wrapper sphere (medium lives only inside it).
        ray.max = sphereExitT(ray, ctx.medium.boundCenter, ctx.medium.boundRadius);
        continue;
      }

      // Shade opaque surfaces only (Null = transparent).
      if (bt != BsdfType::Null) {
        RGB_F32 albedo = lambertianAlbedo();
        f32 ndl = mps::fabsf(mps::dot(hr.ng, sunDir));
        emission += throughput * albedo * (ambient + sun * ndl);
        throughput = throughput * bs.weight;
      }

      ray.o = hr.p + hr.ng * ctx.settings.eps;  // step outside
      ray.dir = bs.wi;
      ray.min = ctx.settings.nearEps;
      ray.max = 1e30f;
    }

    return emission;
  }

}
