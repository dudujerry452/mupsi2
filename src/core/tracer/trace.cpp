#include "core/tracer/trace.h"

#include "core/bsdf/bsdf.h"
#include "core/gp/colorfield.h"
#include "core/gp/gradient_nee.h"
#include "core/light/light.h"
#include "core/math/random.h"

namespace mps::core {

  namespace {

    // GPIS surface color by the selected texturing approach.
    MPHD MPINL RGB_F32 colorSample(const vec3f& p, const Medium& m) {
      const ColorMethod cm = ColorMethod(m.colorMethod);
      if (cm == ColorMethod::Voxel && !m.colorGrid.empty()) return m.colorGrid.eval(p);
      const vec3f mn = m.colorGrid.origin;
      const vec3f mx = m.colorGrid.origin +
                       vec3f{f32(m.colorGrid.res), f32(m.colorGrid.res), f32(m.colorGrid.res)} *
                       m.colorGrid.cellSize;
      if (cm == ColorMethod::Surface) return fieldColor(p, mn, mx);
      if (cm == ColorMethod::Kernel) return kernelColor(p, mn, mx);
      return lambertianAlbedo();
    }

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
                                   const RGB_F32& albedo,
                                   const SceneDevice& scene, const Medium& m,
                                   const GPConditioningState& c, u32 seed,
                                   bool inMedium, Random& rng) {
      if (scene.numLights == 0) return {0.0f, 0.0f, 0.0f};
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
            const f32 pdfL2 = lightDirectPdf(p, L2, wi);
            const f32 wB = powerHeuristic(pdfB, pdfL2);
            L += (albedo * cosB) * (L2.emission * sh / pdfB) * wB;
          }
        }
      }

      return L;
    }

    // Pdf of sampling direction `ws` from the light-sampling strategy (sum over
    // lights of pick-prob × spherical-cap pdf). Used for the UNI-side MIS weight.
    MPHD f32 lightDirectionPdf(const vec3f& p, const vec3f& ws, const SceneDevice& scene) {
      if (scene.numLights == 0) return 0.0f;
      if (scene.numLights == 1) return lightDirectPdf(p, scene.lights[0], ws);
      f32 total = 0.0f;
      for (u32 i = 0; i < scene.numLights; ++i) total += approximateRadiance(p, scene.lights[i]);
      f32 pdf = 0.0f;
      if (total <= 0.0f) {
        for (u32 i = 0; i < scene.numLights; ++i) pdf += lightDirectPdf(p, scene.lights[i], ws);
        return pdf / f32(scene.numLights);
      }
      for (u32 i = 0; i < scene.numLights; ++i)
        pdf += (approximateRadiance(p, scene.lights[i]) / total) * lightDirectPdf(p, scene.lights[i], ws);
      return pdf;
    }

    // NDF in direction space (paper §6 / Eq. 25): f = p(g_xy) / |J|, with the
    // solid-angle→gradient Jacobian |J| = 4 cosθ |n·d| / λ².
    MPHD MPINL f32 phaseEval(f32 pGxy, f32 lambda, f32 cosTheta, f32 nd) {
      const f32 J = 4.0f * cosTheta * mps::fabsf(nd) / (lambda * lambda);
      return (J > 0.0f) ? pGxy / J : 0.0f;
    }

    // §6 NDF next-event estimation for a delta-BSDF (specular microsurface) GPIS.
    // Only valid for isotropic kernels + Renewal/RenewalHalf+/Renewal+; the caller
    // gates this with `if constexpr (SEKernel::isotropic)` and a gpMode check.
    MPHD RGB_F32 specularGpNee(const MediumSample& ms, const vec3f& d,
                               const SceneDevice& scene, const Medium& m,
                               const GPConditioningState& c, u32 seed, bool inMedium,
                               Random& rng) {
      if (scene.numLights == 0) return {0.0f, 0.0f, 0.0f};

      const f32 gz  = mps::dot(ms.gradNoise, d);
      const f32 var = gxyVariance(m);

      f32 pickWeight;
      const i32 li = chooseLight(scene.lights, scene.numLights, ms.p, rng, pickWeight);
      if (li < 0) return {0.0f, 0.0f, 0.0f};

      LightSample ls;
      if (!sampleLightDirect(ms.p, scene.lights[li], rng, ls)) return {0.0f, 0.0f, 0.0f};

      const vec3f wo = -d;
      vec3f gNee, n;
      f32 lambda;
      if (!toGxy(wo, ls.d, d, gz, ms.gradMean, gNee, n, lambda)) return {0.0f, 0.0f, 0.0f};

      const vec3f gxy  = gNee - gz * d;
      const f32 gxySq  = mps::dot(gxy, gxy);
      const f32 pGxy   = evalGxyPdf(gxySq, var);
      const f32 cosTh  = mps::dot(n, ls.d);
      const f32 nd     = mps::dot(n, d);
      const f32 f      = phaseEval(pGxy, lambda, cosTh, nd);
      // Guard degenerate/huge Jacobians (grazing half-vector → λ² overflow → f = inf
      // → inf·0 = NaN). NaN would poison the whole pixel's emission.
      if (!(f > 0.0f) || f > 1e12f || cosTh <= 0.0f) return {0.0f, 0.0f, 0.0f};

      // Shadow ray.
      Ray sr;
      sr.o = ms.p + n * EPS_SURFACE; sr.dir = ls.d; sr.min = EPS_NEAR; sr.max = ls.dist * 0.99f;
      RGB_F32 trans{1.0f, 1.0f, 1.0f};
      if (inMedium) {
        trans = transmittance(sr, m, c, seed);
        if (trans.r <= 0.0f && trans.g <= 0.0f && trans.b <= 0.0f) return {0.0f, 0.0f, 0.0f};
      }
      if (sceneOccluded(sr, scene, u32(li))) return {0.0f, 0.0f, 0.0f};

      // Reference volumeLightSample: lightF = f·e/p_ω, MIS = powerHeuristic(p_ω, f).
      const f32 pOmega = ls.pdf;
      const f32 w = (pOmega * pOmega) / (pOmega * pOmega + f * f);
      const RGB_F32 e = scene.lights[li].emission;
      return e * (f * w / pOmega) * pickWeight * trans;
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
      // Per-bounce medium seed: SingleRealization keeps ONE fixed realization for
      // the whole image (a constant global seed, independent of pixel/spp);
      // Renewal/Renewal+ give each ray segment a fresh, independent realization
      // (matches the original mupsi's make_seed(pixel, spp, seed, bounce)).
      const u32 mSeed = (ctx.medium.gpMode == GPMode::SingleRealization)
                        ? 42u : make_seed(seed, u32(bounce));

      // ——— ray marching branch: inside the wrapped medium ———
      if (inMedium) {
        MediumSample ms;
        if (sampleDistance(ray, ctx.medium, cond, mSeed, ms)) {
          const BsdfType gbt = BsdfType(ctx.medium.surfaceBsdf);

          if (gbt == BsdfType::Lambertian) {
            // Diffuse GPIS: NEE + ambient.
            const RGB_F32 albedo = colorSample(ms.p, ctx.medium);
            emission += throughput * (ambient * albedo +
                                      nextEventEstimate(ms.p, ms.normal, -ray.dir, albedo,
                                                        ctx.scene, ctx.medium, cond, mSeed, true, rng));
          } else if (gbt == BsdfType::Specular) {
            // §6 NDF next-event estimation for the delta microsurface. Compile-time
            // gated on an isotropic kernel; runtime gated on a memory model that has
            // a normal distribution (Renewal/RenewalHalf+/Renewal+, not Single).
            if constexpr (SEKernel::isotropic) {
              if (ctx.medium.gpMode != GPMode::SingleRealization) {
                emission += throughput * specularGpNee(ms, ray.dir, ctx.scene, ctx.medium,
                                                       cond, mSeed, true, rng);
              }
            }
          }

          // Scatter: mirror about the GP normal (specular) or cosine hemisphere
          // (Lambertian).
          BsdfSample bs = bsdfSample(gbt, -ray.dir, ms.normal, {0.0f, 0.0f},
                                     rng.next1D(), rng.next1D());

          if (gbt == BsdfType::Specular) {
            // UNI-side MIS weight (power heuristic): scale the recursive mirror
            // contribution by f²/(p_ω² + f²), where f is the NDF in direction
            // space and p_ω is the light-sampling pdf of the scattered direction.
            if constexpr (SEKernel::isotropic) {
              if (ctx.medium.gpMode != GPMode::SingleRealization) {
                const f32 gz = mps::dot(ms.gradNoise, ray.dir);
                const vec3f gxy = ms.gradNoise - gz * ray.dir;
                const f32 pGxy = evalGxyPdf(mps::dot(gxy, gxy), gxyVariance(ctx.medium));
                const vec3f G = ms.gradMean + ms.gradNoise;   // unconditioned gradient
                const f32 lambdaU = mps::norm(G);
                const vec3f nU = (lambdaU > 1e-6f) ? G * (1.0f / lambdaU) : ms.normal;
                const f32 fU = phaseEval(pGxy, lambdaU, mps::dot(nU, bs.wi), mps::dot(nU, ray.dir));
                const f32 pOmegaU = lightDirectionPdf(ms.p, bs.wi, ctx.scene);
                const f32 denom = fU * fU + pOmegaU * pOmegaU;
                // Degenerate (both pdfs ~0): keep the UNI recursive contribution
                // (it carries the envmap / infinite lights, not the area-light NEE).
                const f32 wUni = (denom > 0.0f) ? (fU * fU) / denom : 1.0f;
                throughput = throughput * wUni;
              }
            }
            wasSpecular = true;
          } else {
            throughput = throughput * bs.weight;
            wasSpecular = false;
          }

          const u32 condSeed = (ctx.medium.gpMode == GPMode::SingleRealization)
                               ? 42u : make_seed(seed, u32(bounce) + 1);
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
        const RGB_F32 albedo = colorSample(hr.p, ctx.medium);
        if (hr.lightId != NO_LIGHT && wasSpecular) {
          // Directly viewing a light through a specular path: emit it (NEE at the
          // previous vertex already counted it for diffuse bounces).
          emission += throughput * ctx.scene.lights[hr.lightId].emission;
        } else {
          emission += throughput * (ambient * albedo +
                                    nextEventEstimate(hr.p, hr.ng, -ray.dir, albedo,
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
