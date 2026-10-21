#pragma once

#include "core/math/math_util.h"

namespace mps {

  // Orthonormal basis built from a normal ([Duff et al. 2017]).
  struct TangentFrame {
    vec3f n, t, b;

    MPHD TangentFrame() : n{0.0f, 0.0f, 1.0f}, t{1.0f, 0.0f, 0.0f}, b{0.0f, 1.0f, 0.0f} {}

    MPHD explicit TangentFrame(const vec3f& normal) {
      n = mps::normalize(normal);
      f32 sign = n.z >= 0.0f ? 1.0f : -1.0f;
      const f32 a = -1.0f / (sign + n.z);
      const f32 bv = n.x * n.y * a;
      t = mps::normalize(vec3f{1.0f + sign * n.x * n.x * a, sign * bv, -sign * n.x});
      b = mps::normalize(vec3f{bv, sign + n.y * n.y * a, -n.y});
    }

    MPHD vec3f toLocal(const vec3f& p) const {
      return {mps::dot(t, p), mps::dot(b, p), mps::dot(n, p)};
    }
    MPHD vec3f toGlobal(const vec3f& p) const {
      return t * p.x + b * p.y + n * p.z;
    }
  };

  // Cosine-weighted hemisphere sample, in local frame (z = normal).
  MPHD MPINL vec3f cosineSampleHemisphere(const vec2f xi) {
    f32 phi = mps::two_pi * xi.x;
    f32 cosTheta = mps::sqrtf(1.0f - xi.y);
    f32 sinTheta = mps::sqrtf(xi.y);
    return {sinTheta * mps::cosf(phi), sinTheta * mps::sinf(phi), cosTheta};
  }
  MPHD MPINL f32 cosineHemispherePdf(f32 cosTheta) { return cosTheta * mps::inv_pi; }

  // Uniform sample on the unit sphere.
  MPHD MPINL vec3f uniformSampleSphere(const vec2f xi) {
    f32 z = 1.0f - 2.0f * xi.x;
    f32 r = mps::sqrtf(mps::maxf(0.0f, 1.0f - z * z));
    f32 phi = mps::two_pi * xi.y;
    return {r * mps::cosf(phi), r * mps::sinf(phi), z};
  }

  // Uniform sample in a spherical cap (z in [cosThetaMax, 1]).
  MPHD MPINL vec3f uniformSampleCap(const vec2f xi, f32 cosThetaMax) {
    f32 phi = xi.x * mps::two_pi;
    f32 z = xi.y * (1.0f - cosThetaMax) + cosThetaMax;
    f32 r = mps::sqrtf(mps::maxf(0.0f, 1.0f - z * z));
    return {mps::cosf(phi) * r, mps::sinf(phi) * r, z};
  }
  MPHD MPINL f32 uniformCapPdf(f32 cosThetaMax) {
    return 1.0f / (mps::two_pi * (1.0f - cosThetaMax));
  }

}
