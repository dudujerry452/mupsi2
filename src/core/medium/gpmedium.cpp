#include "core/medium/gpmedium.h"

#include "core/gp/gpnoise.h"
#include "core/gp/meanfunction.h"
#include "core/math/random.h"

namespace mps::core {

  namespace {

    MPHD MPINL SEKernel       kernelOf(const Medium& m) { return {m.sigma, m.lengthScale, m.kernelRadius}; }
    MPHD MPINL SparseConvNoise noiseOf(const Medium& m) { return {kernelOf(m), m.impulseDensity, m.kernelRadius}; }

    MPHD MPINL f32 meanEval(const Medium& m, const vec3f& p) {
      if (m.meanType == MeanType::Mesh) return m.meshMean.eval(p);
      return SphereMean{m.meanCenter, m.meanRadius}.eval(p);
    }
    MPHD MPINL vec3f meanGrad(const Medium& m, const vec3f& p) {
      if (m.meanType == MeanType::Mesh) return m.meshMean.gradient(p);
      return SphereMean{m.meanCenter, m.meanRadius}.gradient(p);
    }

    MPHD MPINL bool signPos(f32 x) { return x > 0.0f; }

  } // namespace

  // f = μ + ψ
  MPHD f32 gpEval(const vec3f& p, u32 seed, const Medium& m) {
    return gpEvalGrad(p, seed, m).x;
  }

  MPHD f32 gpEvalCond(const vec3f& p, u32 seed, const Medium& m, const GPConditioningState& c) {
    return gpEvalGradCond(p, seed, m, c).x;
  }

  // {f, ∇f}, un-conditioned.
  MPHD vec4f gpEvalGrad(const vec3f& p, u32 seed, const Medium& m) {
    SparseConvNoise noise = noiseOf(m);

    f32 mu = meanEval(m, p);
    vec3f mug = meanGrad(m, p);
    vec4f psi = noise.eval(p, seed);   // {ψ, ∇ψ}

    return {mu + psi.x, mug.x + psi.y, mug.y + psi.z, mug.z + psi.w};
  }

  // {f_cond, ∇f_cond} with the Renewal+ correction
  //   ψ_cond(x) = ψ(x) + α·h(C,x) + β·∇h(C,x)
  //   ∇ψ_cond(x) = ∇ψ(x) + α·∇h(C,x) + H_h(C,x)·β
  MPHD vec4f gpEvalGradCond(const vec3f& p, u32 seed, const Medium& m, const GPConditioningState& c) {
    vec4f f = gpEvalGrad(p, seed, m);

    if (c.active) {
      SEKernel k = kernelOf(m);
      f32   hv = k.h(c.C, p);             // h(C,x)
      vec3f hg = k.hGrad(c.C, p);         // ∇h(C,x)
      vec3f H  = k.hHessianVec(c.C, p, c.beta);  // H_h(C,x)·β

      f.x += c.alpha * hv + mps::dot(c.beta, hg);
      f.y += c.alpha * hg.x + H.x;
      f.z += c.alpha * hg.y + H.y;
      f.w += c.alpha * hg.z + H.z;
    }

    return f;
  }

  MPHD bool gpSampleDistance(const Ray& ray, const Medium& m, const GPConditioningState& c,
                             u32 seed, MediumSample& out) {
    f32 nt = ray.min;
    f32 ft = mps::minf(ray.max, 2000.0f);

    const i32 minDepth = 100;
    const f32 maxDt   = 0.5f;
    f32 dt = mps::minf(maxDt, (ft - nt) / f32(minDepth));

    f32 t = nt;
    f32 fPrev = gpEvalCond(ray.o, seed, m, c);
    bool hit = false;

    while (t < ft) {
      vec3f p = ray.at(t);
      f32 fC = gpEvalCond(p, seed, m, c);

      // Sign flip across a finite gap → zero crossing between (t-dt, t].
      if (signPos(fC) != signPos(fPrev) && mps::fabsf(fPrev - fC) > 1e-6f) {
        // Linear-interpolated candidate, stepped back toward the crossing.
        f32 factor = fPrev / (fPrev - fC);
        do {
          f32 tTest = mps::lerpf(t - dt, t, factor);
          f32 fTest = gpEvalCond(ray.at(tTest), seed, m, c);
          if (signPos(fTest) == signPos(fPrev)) break;
          factor *= 0.9f;
        } while (factor >= 0.01f);
        hit = true;
      }

      if (hit) {
        out.exited = false;
        out.t = t;
        out.p = ray.at(t);
        vec4f f = gpEvalGradCond(out.p, seed, m, c);
        out.normal = mps::normalize(vec3f{f.y, f.z, f.w});
        return true;
      }

      fPrev = fC;
      t += dt;
    }

    return false;
  }

  MPHD RGB_F32 gpTransmittance(const Ray& ray, const Medium& m, const GPConditioningState& c, u32 seed) {
    f32 nt = ray.min;
    f32 ft = mps::minf(ray.max, 2000.0f);

    const i32 minDepth = 100;
    const f32 maxDt   = 0.5f;
    f32 dt = mps::minf(maxDt, (ft - nt) / f32(minDepth));

    f32 t = nt;
    f32 fPrev = gpEvalCond(ray.o, seed, m, c);

    while (t < ft) {
      vec3f p = ray.at(t);
      f32 fC = gpEvalCond(p, seed, m, c);

      if (signPos(fC) != signPos(fPrev) && mps::fabsf(fPrev - fC) > 1e-6f)
        return {0.0f, 0.0f, 0.0f}; // opaque

      fPrev = fC;
      t += dt;
    }

    return {1.0f, 1.0f, 1.0f}; // transparent
  }

  // Renewal+: pin f(C)=0 and match ∇f(C)=n at the hit point.
  MPHD GPConditioningState gpSampleCondition(const MediumSample& s, const Medium& m, u32 seed) {
    GPConditioningState c;
    c.active = false;

    if (m.gpMode == GPMode::RenewalPlus && !s.exited) {
      SparseConvNoise noise = noiseOf(m);
      SEKernel k = kernelOf(m);

      vec4f psi = noise.eval(s.p, seed);  // {ψ, ∇ψ}

      c.active = true;
      c.C = s.p;
      c.alpha = -meanEval(m, s.p) - psi.x;                                // pin f(C)=0
      c.beta  = k.oneOverSecondDerivative() *                             // match ∇f(C)=n
                (s.normal - meanGrad(m, s.p) - vec3f{psi.y, psi.z, psi.w});
    }

    return c;
  }

}
