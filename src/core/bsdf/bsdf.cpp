#include "core/bsdf/bsdf.h"

namespace mps::core {

  MPHD BsdfSample bsdfSample(BsdfType type, const vec3f& wo, const vec3f& n, const vec2f& uv,
                             f32 u1, f32 u2) {
    (void)uv;
    BsdfSample s;

    if (type == BsdfType::Null) {
      s.wi = -wo;
      s.pdf = 1.0f;
      s.weight = {1.0f, 1.0f, 1.0f};
      return s;
    }

    if (type == BsdfType::Specular) {
      // Perfect mirror: reflect the incoming direction about n. Delta lobe
      // (pdf is a point mass; eval/pdf is the Fresnel-free albedo 1).
      s.wi = -wo + 2.0f * mps::dot(wo, n) * n;
      s.pdf = 1.0f;
      s.weight = {1.0f, 1.0f, 1.0f};
      return s;
    }

    // Lambertian: cosine-weighted hemisphere around n.
    TangentFrame frame(n);
    vec3f localWi = cosineSampleHemisphere({u1, u2});
    s.wi = frame.toGlobal(localWi);
    f32 cosTheta = localWi.z; // dot(localWi, +z)
    s.pdf = cosineHemispherePdf(cosTheta);
    s.weight = lambertianAlbedo(); // eval / pdf == albedo
    return s;
  }

  MPHD RGB_F32 bsdfEval(BsdfType type, const vec3f& wo, const vec3f& wi, const vec3f& n, const vec2f& uv) {
    (void)uv;
    if (type == BsdfType::Null) {
      return (mps::dot(wi, -wo) > 0.999f) ? RGB_F32{1.0f, 1.0f, 1.0f} : RGB_F32{0.0f, 0.0f, 0.0f};
    }
    if (type == BsdfType::Specular) {
      return {0.0f, 0.0f, 0.0f};  // delta: zero measure for any non-mirror direction
    }
    f32 cosTheta = mps::maxf(0.0f, mps::dot(wi, n));
    return lambertianAlbedo() * (cosTheta * mps::inv_pi);
  }

  MPHD f32 bsdfPdf(BsdfType type, const vec3f& wo, const vec3f& wi, const vec3f& n) {
    if (type == BsdfType::Null) {
      return (mps::dot(wi, -wo) > 0.999f) ? 1.0f : 0.0f;
    }
    if (type == BsdfType::Specular) {
      return 0.0f;  // delta: infinite density along the mirror direction
    }
    return cosineHemispherePdf(mps::maxf(0.0f, mps::dot(wi, n)));
  }

}
