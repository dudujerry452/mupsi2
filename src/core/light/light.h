#pragma once

#include "core/basic/types.h"
#include "core/math/math_util.h"
#include "core/math/sample.h"
#include "core/math/random.h"

namespace mps::core {

  // Spherical area light with constant emission — the part of Tungsten's Sphere
  // emitter we need (position/radius/radiance). The light is also a regular
  // sphere primitive in the scene (so it can be intersected/blocked).
  struct AreaLight {
    vec3f center{0.0f, 0.0f, 0.0f};
    f32   radius = 1.0f;
    RGB_F32 emission{1.0f, 1.0f, 1.0f};
  };

  struct LightSample {
    vec3f d{0.0f, 0.0f, 1.0f};   // normalized dir shade point -> sampled point
    f32   dist = 0.0f;
    f32   pdf  = 0.0f;
    RGB_F32 radiance{0.0f, 0.0f, 0.0f};  // emission / pdf
    u32   lightId = 0;
  };

  // Sample a direction to a spherical area light: uniform cap around the
  // light's angular extent (Tungsten Sphere::sampleDirect).
  MPHD MPINL bool sampleLightDirect(const vec3f& p, const AreaLight& L, Random& rng, LightSample& out) {
    vec3f toCenter = L.center - p;
    const f32 d = mps::norm(toCenter);
    const f32 C = d * d - L.radius * L.radius;
    if (C <= 0.0f) return false;   // shade point inside the light

    toCenter = toCenter / d;
    const f32 cosTheta = mps::sqrtf(C) / d;

    const vec3f local = uniformSampleCap(vec2f{rng.next1D(), rng.next1D()}, cosTheta);
    const f32 B = d * local.z;
    const f32 det = mps::sqrtf(mps::maxf(B * B - C, 0.0f));
    out.dist = B - det;
    out.d = TangentFrame(toCenter).toGlobal(local);
    out.pdf = uniformCapPdf(cosTheta);
    out.radiance = L.emission / out.pdf;
    return true;
  }

  // Pdf of sampling `dir` from p toward the light (MIS counterpart).
  MPHD MPINL f32 lightDirectPdf(const vec3f& p, const AreaLight& L) {
    const f32 d = mps::norm(L.center - p);
    const f32 cosTheta = mps::sqrtf(mps::maxf(d * d - L.radius * L.radius, 0.0f)) / d;
    return uniformCapPdf(cosTheta);
  }

  // Approximate radiance for light importance sampling (Tungsten approximateRadiance).
  MPHD MPINL f32 approximateRadiance(const vec3f& p, const AreaLight& L) {
    const f32 d = mps::norm(L.center - p);
    const f32 em = mps::maxf(L.emission.r, mps::maxf(L.emission.g, L.emission.b));
    if (d <= L.radius) return mps::two_pi * em;   // inside: full sphere
    const f32 cosTheta = mps::sqrtf(d * d - L.radius * L.radius) / d;
    return mps::two_pi * (1.0f - cosTheta) * em;  // cap solid angle * emission
  }

  // Power heuristic (Veach) for MIS between two sampling strategies.
  MPHD MPINL f32 powerHeuristic(f32 pdf0, f32 pdf1) {
    const f32 p0 = pdf0 * pdf0, p1 = pdf1 * pdf1;
    const f32 d = p0 + p1;
    return d > 0.0f ? p0 / d : 0.5f;
  }

  // Importance-sample a light by approximate radiance. Returns the light index
  // and sets `pickWeight` = 1 / P(chosen light) (multiply the estimator by it).
  // Two passes, no variable-length scratch — device friendly.
  MPHD MPINL i32 chooseLight(const AreaLight* lights, u32 numLights, const vec3f& p, Random& rng, f32& pickWeight) {
    if (numLights == 0) { pickWeight = 1.0f; return -1; }
    if (numLights == 1) { pickWeight = 1.0f; return 0; }

    f32 total = 0.0f;
    for (u32 i = 0; i < numLights; ++i) total += approximateRadiance(p, lights[i]);

    if (total <= 0.0f) {   // degenerate: uniform fallback
      pickWeight = f32(numLights);
      return i32(rng.nextI() % numLights);
    }

    f32 t = rng.next1D() * total;
    for (u32 i = 0; i < numLights; ++i) {
      const f32 r = approximateRadiance(p, lights[i]);
      if (t < r || i == numLights - 1) {
        pickWeight = total / r;
        return i32(i);
      }
      t -= r;
    }
    pickWeight = 1.0f;
    return 0;
  }

}
