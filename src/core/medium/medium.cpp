#include "core/medium/medium.h"

#include "core/medium/sdfmedium.h"
#include "core/medium/gpmedium.h"

namespace mps::core {

  MPHD bool sampleDistance(const Ray& ray, const Medium& m, const GPConditioningState& c,
                           u32 seed, MediumSample& out) {
    if (m.type == MediumType::SDF) return sdfSampleDistance(ray, m, c, seed, out);
    return gpSampleDistance(ray, m, c, seed, out);
  }

  MPHD RGB_F32 transmittance(const Ray& ray, const Medium& m, const GPConditioningState& c, u32 seed) {
    if (m.type == MediumType::SDF) return sdfTransmittance(ray, m, c, seed);
    return gpTransmittance(ray, m, c, seed);
  }

  MPHD GPConditioningState sampleCondition(const MediumSample& s, const Medium& m, u32 seed) {
    return gpSampleCondition(s, m, seed);
  }

}
