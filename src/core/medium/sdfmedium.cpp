#include "core/medium/sdfmedium.h"

#include "core/gp/meanfunction.h"

namespace mps::core {

  namespace {
    MPHD MPINL SphereMean meanOf(const Medium& m) { return {m.meanCenter, m.meanRadius}; }
  }

  MPHD bool sdfSampleDistance(const Ray& ray, const Medium& m, const GPConditioningState& c,
                              u32 seed, MediumSample& out) {
    (void)c; (void)seed;

    SphereMean mean = meanOf(m);
    f32 nt = ray.min;
    f32 ft = mps::minf(ray.max, 2000.0f);

    const f32 minStep = 1e-3f;
    const i32 maxIter = 512;

    f32 t = nt;
    f32 fPrev = mean.eval(ray.o);

    for (i32 iter = 0; iter < maxIter && t < ft; ++iter) {
      vec3f p = ray.at(t);
      f32 tPrev = t;
      f32 fC = mean.eval(p);

      // Sign change → bisection refine.
      if ((fPrev > 0.0f) != (fC > 0.0f)) {
        f32 lo = tPrev, hi = t;
        f32 fLo = fPrev;
        for (i32 b = 0; b < 16; ++b) {
          f32 mid = 0.5f * (lo + hi);
          f32 fMid = mean.eval(ray.at(mid));
          if ((fLo > 0.0f) == (fMid > 0.0f)) { lo = mid; fLo = fMid; }
          else { hi = mid; }
        }

        out.exited = false;
        out.t = hi;
        out.p = ray.at(hi);
        out.normal = mean.gradient(out.p);
        return true;
      }

      // Sphere tracing: |f| is a safe distance to the surface.
      f32 step = mps::maxf(mps::fabsf(fC), minStep);
      fPrev = fC;
      t += step;
    }

    return false;
  }

  MPHD RGB_F32 sdfTransmittance(const Ray& ray, const Medium& m, const GPConditioningState& c, u32 seed) {
    (void)c; (void)seed;

    SphereMean mean = meanOf(m);
    f32 nt = ray.min;
    f32 ft = mps::minf(ray.max, 2000.0f);

    const f32 minStep = 1e-3f;
    const i32 maxIter = 512;

    f32 t = nt;
    f32 fPrev = mean.eval(ray.o);

    for (i32 iter = 0; iter < maxIter && t < ft; ++iter) {
      vec3f p = ray.at(t);
      f32 fC = mean.eval(p);

      if ((fPrev > 0.0f) != (fC > 0.0f)) return {0.0f, 0.0f, 0.0f};

      f32 step = mps::maxf(mps::fabsf(fC), minStep);
      fPrev = fC;
      t += step;
    }

    return {1.0f, 1.0f, 1.0f};
  }

}
