#include "core/medium/gpmedium.h"

#include "core/gp/gpnoise.h"
#include "core/gp/meanfunction.h"
#include "core/math/random.h"
#include "core/tracer/constant.h"

namespace mps::core {

  namespace {

    MPHD MPINL SEKernel       kernelOf(const Medium& m) { return {m.sigma, m.lengthScale, m.kernelRadius}; }
    MPHD MPINL SparseConvNoise noiseOf(const Medium& m) { return {kernelOf(m), m.impulseDensity, m.kernelRadius}; }

    MPHD MPINL f32 meanEval(const Medium& m, const vec3f& p) {
      if (m.meanType == MeanType::Mesh) return m.meshMean.eval(p);
      if (m.meanType == MeanType::Voxel) return m.voxelMean.eval(p);
      return SphereMean{m.meanCenter, m.meanRadius}.eval(p);
    }
    MPHD MPINL vec3f meanGrad(const Medium& m, const vec3f& p) {
      if (m.meanType == MeanType::Mesh) return m.meshMean.gradient(p);
      if (m.meanType == MeanType::Voxel) return m.voxelMean.gradient(p);
      return SphereMean{m.meanCenter, m.meanRadius}.gradient(p);
    }
    MPHD MPINL vec4f meanEvalGrad(const Medium& m, const vec3f& p) {
      if (m.meanType == MeanType::Mesh) return m.meshMean.evalGrad(p);
      if (m.meanType == MeanType::Voxel) return m.voxelMean.evalGrad(p);
      return SphereMean{m.meanCenter, m.meanRadius}.evalGrad(p);
    }

    MPHD MPINL bool signPos(f32 x) { return x > 0.0f; }

  } // namespace

  // Compile-time dispatch: 1D noise is valid only for isotropic kernels AND a
  // memory model that does NOT require a single fixed 3D realization
  // (paper line 477: Renewal / Renewal Half+ / Renewal+). The 1D branch is
  // compiled out entirely when the kernel is anisotropic.
  MPHD MPINL bool use1DNoise(const Medium& m) {
    if constexpr (SEKernel::isotropic) return m.gpMode != GPMode::SingleRealization;
    return false;
  }

  // Unconditioned noise {ψ, ∇ψ} at p = o + t·d. The 1D path samples the value
  // and the directional derivative g_z from a 1D sparse convolution along the
  // ray, then assembles the perpendicular g_xy ~ N(0, σ²/L²) in the tangent
  // plane (paper §6.2.1: g_xy ⊥ g_z, isotropic 2D Gaussian).
  MPHD vec4f gpNoiseGrad(const vec3f& p, const vec3f& d, f32 t, u32 seed, const Medium& m) {
    SparseConvNoise noise = noiseOf(m);

    if (!use1DNoise(m)) return noise.eval(p, seed);

    const vec2f n1 = noise.eval1D(t, seed);   // {ψ, ∂ψ/∂t}
    const f32   L  = m.lengthScale.x;         // isotropic: all components equal

    // Orthonormal tangent frame {t1, t2, d} with d as the z axis.
    const vec3f ref = (mps::fabsf(d.x) > 0.9f) ? vec3f{0.0f, 1.0f, 0.0f} : vec3f{1.0f, 0.0f, 0.0f};
    const vec3f t1 = mps::normalize(mps::cross(d, ref));
    const vec3f t2 = mps::cross(d, t1);

    Random rng(make_seed(seed, 0x9E3779B9u)); // stream distinct from the 1D cell hash
    const f32 gxyStd = m.sigma / L;           // Var(∂ψ/∂x) = σ²/L²
    const f32 gx = rng.standard_normal() * gxyStd;
    const f32 gy = rng.standard_normal() * gxyStd;

    const vec3f g = d * n1.y + t1 * gx + t2 * gy;
    return {n1.x, g.x, g.y, g.z};
  }

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

    vec4f mg  = meanEvalGrad(m, p);    // {μ, ∇μ} in one query
    vec4f psi = noise.eval(p, seed);   // {ψ, ∇ψ}

    return {mg.x + psi.x, mg.y + psi.y, mg.z + psi.z, mg.w + psi.w};
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

  // Value-only, along ray p = o + t·d (used by the march). Conditioning is a
  // single full-3D splat (cheap), independent of the 1D/3D noise choice.
  MPHD f32 gpEvalCondAlong(const vec3f& o, const vec3f& d, f32 t, u32 seed,
                           const Medium& m, const GPConditioningState& c) {
    const vec3f p = o + d * t;
    f32 f = meanEval(m, p) + gpNoiseGrad(p, d, t, seed, m).x;

    if (c.active) {
      SEKernel k = kernelOf(m);
      f += c.alpha * k.h(c.C, p) + mps::dot(c.beta, k.hGrad(c.C, p));
    }
    return f;
  }

  // {f, ∇f} with conditioning, along ray (used at the hit point for the normal).
  MPHD vec4f gpEvalGradCondAlong(const vec3f& o, const vec3f& d, f32 t, u32 seed,
                                 const Medium& m, const GPConditioningState& c) {
    const vec3f p = o + d * t;
    vec4f mg  = meanEvalGrad(m, p);          // {μ, ∇μ}
    vec4f psi = gpNoiseGrad(p, d, t, seed, m); // {ψ, ∇ψ}
    vec4f f{mg.x + psi.x, mg.y + psi.y, mg.z + psi.z, mg.w + psi.w};

    if (c.active) {
      SEKernel k = kernelOf(m);
      f32   hv = k.h(c.C, p);
      vec3f hg = k.hGrad(c.C, p);
      vec3f H  = k.hHessianVec(c.C, p, c.beta);

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
    f32 ft = mps::minf(ray.max, m.sceneScale * MARCH_FAR_CLAMP_FRAC);

    const f32 maxDt   = m.sceneScale * MARCH_MAX_DT_FRAC;
    const f32 signEps = m.sceneScale * MARCH_SIGN_EPS_FRAC;
    f32 dt = mps::minf(maxDt, (ft - nt) / f32(MARCH_MIN_DEPTH));

    f32 t = nt;
    f32 fPrev = gpEvalCondAlong(ray.o, ray.dir, nt, seed, m, c);
    bool hit = false;

    while (t < ft) {
      f32 fC = gpEvalCondAlong(ray.o, ray.dir, t, seed, m, c);

      // Sign flip across a finite gap → zero crossing between (t-dt, t].
      if (signPos(fC) != signPos(fPrev) && mps::fabsf(fPrev - fC) > signEps) {
        // Linear-interpolated candidate, stepped back toward the crossing.
        f32 factor = fPrev / (fPrev - fC);
        do {
          f32 tTest = mps::lerpf(t - dt, t, factor);
          f32 fTest = gpEvalCondAlong(ray.o, ray.dir, tTest, seed, m, c);
          if (signPos(fTest) == signPos(fPrev)) break;
          factor *= 0.9f;
        } while (factor >= 0.01f);
        hit = true;
      }

      if (hit) {
        out.exited = false;
        out.t = t;
        out.p = ray.at(t);
        vec4f f = gpEvalGradCondAlong(ray.o, ray.dir, t, seed, m, c);
        vec3f grad{f.y, f.z, f.w};
        const f32 glen = mps::norm(grad);
        // Guard against a near-zero total gradient (rare, but a single frozen
        // realization would otherwise produce NaN normals -> fireflies).
        out.normal = (glen > 1e-6f) ? grad * (1.0f / glen) : meanGrad(m, out.p);
        return true;
      }

      fPrev = fC;
      t += dt;
    }

    return false;
  }

  MPHD RGB_F32 gpTransmittance(const Ray& ray, const Medium& m, const GPConditioningState& c, u32 seed) {
    f32 nt = ray.min;
    f32 ft = mps::minf(ray.max, m.sceneScale * MARCH_FAR_CLAMP_FRAC);

    const f32 maxDt   = m.sceneScale * MARCH_MAX_DT_FRAC;
    const f32 signEps = m.sceneScale * MARCH_SIGN_EPS_FRAC;
    f32 dt = mps::minf(maxDt, (ft - nt) / f32(MARCH_MIN_DEPTH));

    f32 t = nt;
    f32 fPrev = gpEvalCondAlong(ray.o, ray.dir, nt, seed, m, c);

    while (t < ft) {
      f32 fC = gpEvalCondAlong(ray.o, ray.dir, t, seed, m, c);

      if (signPos(fC) != signPos(fPrev) && mps::fabsf(fPrev - fC) > signEps)
        return {0.0f, 0.0f, 0.0f}; // opaque

      fPrev = fC;
      t += dt;
    }

    return {1.0f, 1.0f, 1.0f}; // transparent
  }

  // Renewal+: pin f(C)=0 and match ∇f(C)=n at the hit point.
  MPHD GPConditioningState gpSampleCondition(const MediumSample& s, const Medium& m, u32 seed,
                                             const vec3f& rayDir) {
    GPConditioningState c;
    c.active = false;

    if (m.gpMode == GPMode::RenewalPlus && !s.exited) {
      SEKernel k = kernelOf(m);

      // Same noise realization as the march that found this hit: along the march
      // ray (1D for Renewal+, 3D otherwise) at t = s.t.
      vec4f psi = gpNoiseGrad(s.p, rayDir, s.t, seed, m);

      c.active = true;
      c.C = s.p;
      c.alpha = -meanEval(m, s.p) - psi.x;                                // pin f(C)=0
      c.beta  = k.oneOverSecondDerivative() *                             // match ∇f(C)=n
                (s.normal - meanGrad(m, s.p) - vec3f{psi.y, psi.z, psi.w});
    }

    return c;
  }

}
