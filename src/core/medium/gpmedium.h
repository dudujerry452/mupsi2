#pragma once

#include "core/medium/medium.h"

namespace mps::core {

  // GP implicit surface: f(p) = μ(p) + ψ(p), ψ ~ GP(0, κ) via sparse convolution.
  MPHD f32   gpEval      (const vec3f& p, u32 seed, const Medium& m);
  MPHD f32   gpEvalCond  (const vec3f& p, u32 seed, const Medium& m, const GPConditioningState& c);
  MPHD vec4f gpEvalGrad  (const vec3f& p, u32 seed, const Medium& m);                       // {f, ∇f}
  MPHD vec4f gpEvalGradCond(const vec3f& p, u32 seed, const Medium& m, const GPConditioningState& c);

  MPHD bool    gpSampleDistance(const Ray& ray, const Medium& m, const GPConditioningState& c, u32 seed, MediumSample& out);
  MPHD RGB_F32 gpTransmittance (const Ray& ray, const Medium& m, const GPConditioningState& c, u32 seed);
  MPHD GPConditioningState gpSampleCondition(const MediumSample& s, const Medium& m, u32 seed);

}
