#pragma once

#include "core/medium/medium.h"

namespace mps::core {

  // Deterministic SDF medium (sphere tracing), used to validate the medium
  // abstraction before GP noise lands. μ(p) = ‖p - center‖ - radius.
  MPHD bool    sdfSampleDistance(const Ray& ray, const Medium& m, const GPConditioningState& c, u32 seed, MediumSample& out);
  MPHD RGB_F32 sdfTransmittance (const Ray& ray, const Medium& m, const GPConditioningState& c, u32 seed);

}
