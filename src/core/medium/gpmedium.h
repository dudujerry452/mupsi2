#pragma once

#include "core/medium/medium.h"

namespace mps::core {

  // GP implicit surface: f(p) = μ(p) + ψ(p), ψ ~ GP(0, κ) via sparse convolution.
  MPHD f32   gpEval      (const vec3f& p, u32 seed, const Medium& m);
  MPHD f32   gpEvalCond  (const vec3f& p, u32 seed, const Medium& m, const GPConditioningState& c);
  MPHD vec4f gpEvalGrad  (const vec3f& p, u32 seed, const Medium& m);                       // {f, ∇f}
  MPHD vec4f gpEvalGradCond(const vec3f& p, u32 seed, const Medium& m, const GPConditioningState& c);

  // Along-ray evaluators used by ray marching. When the kernel is isotropic and the
  // memory model is Renewal/Renewal+ (not SingleRealization), the value is sampled
  // from a 1D noise along the ray (3 cells) instead of the 3D noise (27 cells); the
  // hit-point gradient assembles the along-ray derivative g_z·d plus an independent
  // perpendicular g_xy ~ N(0, σ²/L²) (paper §6.2.1 / line 477, 520).
  MPHD f32   gpEvalCondAlong    (const vec3f& o, const vec3f& d, f32 t, u32 seed, const Medium& m, const GPConditioningState& c);
  MPHD vec4f gpEvalGradCondAlong(const vec3f& o, const vec3f& d, f32 t, u32 seed, const Medium& m, const GPConditioningState& c);

  MPHD bool    gpSampleDistance(const Ray& ray, const Medium& m, const GPConditioningState& c, u32 seed, MediumSample& out);
  MPHD RGB_F32 gpTransmittance (const Ray& ray, const Medium& m, const GPConditioningState& c, u32 seed);
  MPHD GPConditioningState gpSampleCondition(const MediumSample& s, const Medium& m, u32 seed, const vec3f& rayDir);

}
