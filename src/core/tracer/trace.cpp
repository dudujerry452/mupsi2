#include "core/tracer/trace.h"

#include "core/bsdf/bsdf.h"
#include "core/light/light.h"
#include "core/math/random.h"

namespace mps::core {

  namespace {

    // Closest hit skipping transparent (Null) primitives — walks through the
    // wrapped-medium boundary so a shadow/BSDF ray can reach a light behind it.
    MPHD bool firstOpaqueHit(const Ray& ray, const SceneDevice& s, HitRecord& out) {
      Ray r = ray;
      for (i32 k = 0; k < 8; ++k) {
        HitRecord hr = sceneIntersect(r, s);
        if (!hr.hit) return false;
        if (hr.bsdfId != u16(BsdfType::Null)) { out = hr; return true; }
        r.o   = ray.o + ray.dir * (hr.t + EPS_SURFACE);
        r.min = EPS_NEAR;
      }
      return false;
    }

    // Direct lighting via next-event estimation with MIS (paper Algorithm 4 /
    // Tungsten TraceBase::sampleDirect): the light-sampled and BSDF-sampled
    // strategies are combined with the power heuristic.
    MPHD RGB_F32 nextEventEstimate(const vec3f& p, const vec3f& n, const vec3f& wo,
                                   const SceneDevice& scene, const Medium& m,
                                   const GPConditioningState& c, u32 seed,
                                   bool inMedium, Random& rng) {
      if (scene.numLights == 0) return {0.0f, 0.0f, 0.0f};
      const RGB_F32 albedo = lambertianAlbedo();
      RGB_F32 L{0.0f, 0.0f, 0.0f};

      // —— strategy 1: light sampling ——
      f32 pickWeight;
      const i32 li = chooseLight(scene.lights, scene.numLights, p, rng, pickWeight);
      if (li >= 0) {
        LightSample ls;
        if (sampleLightDirect(p, scene.lights[li], rng, ls)) {
          const f32 cosL = mps::dot(ls.d, n);
          if (cosL > 0.0f) {
            Ray sr;
            sr.o = p + n * EPS_SURFACE; sr.dir = ls.d; sr.min = EPS_NEAR; sr.max = ls.dist * 0.99f;
            RGB_F32 sh{1.0f, 1.0f, 1.0f};
            if (inMedium) {
              sh = transmittance(sr, m, c, seed);
              if (sh.r <= 0.0f && sh.g <= 0.0f && sh.b <= 0.0f) sh = {0.0f, 0.0f, 0.0f};
            }
            if (!sceneOccluded(sr, scene, u32(li)) && (sh.r > 0.0f || sh.g > 0.0f || sh.b > 0.0f)) {
              const f32 pdfB = bsdfPdf(BsdfType::Lambertian, wo, ls.d, n);
              const f32 wL = powerHeuristic(ls.pdf, pdfB);
              L += (albedo * cosL) * ls.radiance * sh * pickWeight * wL;
            }
          }
        }
      }

      // —— strategy 2: BSDF sampling (cosine hemisphere) ——
      {
        const vec3f localWi = cosineSampleHemisphere({rng.next1D(), rng.next1D()});
        const vec3f wi = TangentFrame(n).toGlobal(localWi);
        const f32 cosB = localWi.z;
        const f32 pdfB = cosineHemispherePdf(cosB);
        if (cosB > 0.0f) {
          Ray br;
          br.o = p + n * EPS_SURFACE; br.dir = wi; br.min = EPS_NEAR; br.max = 1e30f;
          RGB_F32 sh{1.0f, 1.0f, 1.0f};
          if (inMedium) sh = transmittance(br, m, c, seed);

          HitRecord hr;
          if ((sh.r > 0.0f || sh.g > 0.0f || sh.b > 0.0f) &&
              firstOpaqueHit(br, scene, hr) && hr.lightId != NO_LIGHT) {
            const AreaLight& L2 = scene.lights[hr.lightId];
            const f32 pdfL2 = lightDirectPdf(p, L2);
            const f32 wB = powerHeuristic(pdfB, pdfL2);
            L += (albedo * cosB) * (L2.emission * sh / pdfB) * wB;
          }
        }
      }

      return L;
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
      // Per-bounce medium seed: SingleRealization keeps one fixed realization for
      // the whole image; Renewal/Renewal+ give each ray segment a fresh, independent
      // realization (matches the original mupsi's make_seed(pixel, spp, seed, bounce)).
      const u32 mSeed = (ctx.medium.gpMode == GPMode::SingleRealization)
                        ? seed : make_seed(seed, u32(bounce));

      // ——— ray marching branch: inside the wrapped medium ———
      if (inMedium) {
        MediumSample ms;
        if (sampleDistance(ray, ctx.medium, cond, mSeed, ms)) {
          const BsdfType gbt = BsdfType(ctx.medium.surfaceBsdf);

          // Direct lighting at the GP surface: diffuse only — a specular (delta)
          // GPIS gets its lighting from the reflected ray, not NEE.
          if (gbt == BsdfType::Lambertian) {
            const RGB_F32 albedo = lambertianAlbedo();
            emission += throughput * (ambient * albedo +
                                      nextEventEstimate(ms.p, ms.normal, -ray.dir,
                                                        ctx.scene, ctx.medium, cond, mSeed, true, rng));
          }

          // Scatter: mirror about the noisy GP normal (specular) or cosine
          // hemisphere (Lambertian).
          BsdfSample bs = bsdfSample(gbt, -ray.dir, ms.normal, {0.0f, 0.0f},
                                     rng.next1D(), rng.next1D());
          throughput = throughput * bs.weight;
          wasSpecular = (gbt == BsdfType::Specular);   // mirror preserves specularness

          const u32 condSeed = (ctx.medium.gpMode == GPMode::SingleRealization)
                               ? seed : make_seed(seed, u32(bounce) + 1);
          cond = sampleCondition(ms, ctx.medium, condSeed, ray.dir);

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
        emission += throughput * background(ray.dir, ctx.envMap);
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
                                    nextEventEstimate(hr.p, hr.ng, -ray.dir,
                                                      ctx.scene, ctx.medium, cond, mSeed, false, rng));
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
