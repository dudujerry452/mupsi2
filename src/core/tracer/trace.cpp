#include "core/tracer/trace.h"

#include "core/bsdf/bsdf.h"
#include "core/light/light.h"
#include "core/math/random.h"

namespace mps::core {

  namespace {

    // Direct lighting via next-event estimation (Tungsten-style light sampling):
    // importance-sample a light by approximate radiance, sample a direction to a
    // point on its area (spherical cap), evaluate the diffuse BRDF, and attenuate
    // by a shadow ray (GP medium transmittance + hard-surface occlusion).
    MPHD RGB_F32 nextEventEstimate(const vec3f& p, const vec3f& n, const SceneDevice& scene,
                                   const Medium& m, const GPConditioningState& c, u32 seed,
                                   bool inMedium, Random& rng) {
      if (scene.numLights == 0) return {0.0f, 0.0f, 0.0f};

      f32 pickWeight;
      const i32 li = chooseLight(scene.lights, scene.numLights, p, rng, pickWeight);
      if (li < 0) return {0.0f, 0.0f, 0.0f};

      LightSample ls;
      if (!sampleLightDirect(p, scene.lights[li], rng, ls)) return {0.0f, 0.0f, 0.0f};

      const f32 cosTheta = mps::dot(ls.d, n);
      if (cosTheta <= 0.0f) return {0.0f, 0.0f, 0.0f};

      const RGB_F32 f = lambertianAlbedo() * cosTheta;   // diffuse shading (albedo·cosθ)

      // Shadow ray: GP medium transmittance, then hard-surface occlusion up to
      // just short of the sampled light point.
      Ray shadowRay;
      shadowRay.o   = p + n * EPS_SURFACE;
      shadowRay.dir = ls.d;
      shadowRay.min = EPS_NEAR;
      shadowRay.max = ls.dist * 0.99f;

      RGB_F32 trans{1.0f, 1.0f, 1.0f};
      if (inMedium) {
        trans = transmittance(shadowRay, m, c, seed);
        if (trans.r <= 0.0f && trans.g <= 0.0f && trans.b <= 0.0f) return {0.0f, 0.0f, 0.0f};
      }
      if (sceneOccluded(shadowRay, scene, u32(li))) return {0.0f, 0.0f, 0.0f};

      return f * ls.radiance * trans * pickWeight;
    }

  } // namespace

  MPHD RGB_F32 trace(const Ray& ray0, const RenderContext& ctx, u32 seed) {
    Ray ray = ray0;
    RGB_F32 emission{0.0f, 0.0f, 0.0f};
    RGB_F32 throughput{1.0f, 1.0f, 1.0f};
    GPConditioningState cond;

    const RGB_F32 ambient{0.03f, 0.03f, 0.04f};

    Random rng(seed);
    bool inMedium = false;     // a path starts in free space; it enters a medium only
                               // after crossing a wrapped (Null, mediumId) boundary
    bool wasSpecular = true;   // camera ray counts as specular: direct light hits emit

    for (i32 bounce = 0; bounce < ctx.settings.maxBounce; ++bounce) {

      // ——— ray marching branch: inside the wrapped medium ———
      if (inMedium) {
        MediumSample ms;
        if (sampleDistance(ray, ctx.medium, cond, seed, ms)) {
          // Hit the GP implicit surface; shade with the gradient as the normal.
          const RGB_F32 albedo = lambertianAlbedo();
          emission += throughput * (ambient * albedo +
                                    nextEventEstimate(ms.p, ms.normal, ctx.scene, ctx.medium,
                                                      cond, seed, true, rng));

          // Scatter (Lambertian), then pin the surface for the next bounce.
          BsdfSample bs = bsdfSample(BsdfType::Lambertian, -ray.dir, ms.normal, {0.0f, 0.0f},
                                     rng.next1D(), rng.next1D());
          throughput = throughput * bs.weight;
          wasSpecular = false;

          cond = sampleCondition(ms, ctx.medium, seed, ray.dir);

          ray.o = ms.p + ms.normal * ctx.settings.eps;
          ray.dir = bs.wi;
          ray.min = ctx.settings.nearEps;
          ray.max = 1e30f;

          // Russian roulette on the path throughput.
          f32 survival = mps::minf(1.0f, mps::maxf(throughput.r, mps::maxf(throughput.g, throughput.b)));
          if (bounce > 2 && rng.next1D() > survival) break;
          if (survival > 0.0f) throughput = throughput / survival;
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

      if (bt == BsdfType::Null) {
        // Transparent boundary: no shading, specularness unchanged.
      } else if (bt == BsdfType::Specular) {
        // Perfect mirror: no NEE (delta lobe); the reflected ray carries the
        // lighting. Specularness is preserved so a direct light hit still emits.
        throughput = throughput * bs.weight;   // {1,1,1}
      } else {
        // Opaque Lambertian surface.
        const RGB_F32 albedo = lambertianAlbedo();
        if (hr.lightId != NO_LIGHT && wasSpecular) {
          // Directly viewing a light through a specular path: emit it (NEE at the
          // previous vertex already counted it for diffuse bounces).
          emission += throughput * ctx.scene.lights[hr.lightId].emission;
        } else {
          emission += throughput * (ambient * albedo +
                                    nextEventEstimate(hr.p, hr.ng, ctx.scene, ctx.medium,
                                                      cond, seed, false, rng));
        }
        throughput = throughput * bs.weight;
        wasSpecular = false;
      }

      ray.o = hr.p + hr.ng * ctx.settings.eps;  // step outside
      ray.dir = bs.wi;
      ray.min = ctx.settings.nearEps;
      ray.max = 1e30f;

      // Russian roulette on the path throughput.
      f32 survival = mps::minf(1.0f, mps::maxf(throughput.r, mps::maxf(throughput.g, throughput.b)));
      if (bounce > 2 && rng.next1D() > survival) break;
      if (survival > 0.0f) throughput = throughput / survival;
    }

    return emission;
  }

}
