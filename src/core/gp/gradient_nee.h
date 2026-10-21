#pragma once

#include "core/gp/kernel.h"
#include "core/medium/medium.h"
#include "core/math/math_util.h"

namespace mps::core {

  // Analytical gradient distribution NEE for a delta-BSDF (specular microsurface)
  // GPIS — paper §6 / Algorithm 4.
  //
  // Key fact (paper §6.2.1): for a stationary ISOTROPIC kernel, the gradient at a
  // hit decomposes as g = (g_xy, g_z) with g_z = the along-ray derivative (fixed by
  // the realization) and g_xy ⊥ g_z, g_xy ~ N(0, σ²/L² · I₂). The GPIS normal
  // n = normalize(∇μ + g) therefore has a closed-form 2D Gaussian distribution in
  // slope space (a generalized Beckmann NDF). This lets us NEXT-EVENT-ESTIMATE a
  // delta micro-BSDF: pick the light direction, solve for the g_xy that reflects
  // to it (TOGXY), and weight by the NDF (EVALGXYPDF) with the solid-angle→gradient
  // Jacobian (TOGXYMEASURE).
  //
  // COMPILE-TIME REQUIREMENT: this is only valid for isotropic kernels
  // (SEKernel::isotropic == true) and a memory model that is NOT SingleRealization
  // (which has a deterministic normal, hence no distribution to sample). Callers
  // gate with `if constexpr (SEKernel::isotropic)` and check gpMode at runtime.

  // Variance of each g_xy component: σ²/L².
  MPHD MPINL f32 gxyVariance(const Medium& m) {
    const f32 L2 = mps::dot(m.lengthScale, m.lengthScale);
    return (m.sigma * m.sigma) / L2;
  }

  // EVALGXYPDF: 2D isotropic Gaussian pdf of g_xy, N(0, var·I₂); takes the
  // squared norm |g_xy|² (g_xy is the component of ∇ψ perpendicular to the ray).
  MPHD MPINL f32 evalGxyPdf(f32 gxySq, f32 var) {
    return (1.0f / (mps::two_pi * var)) * mps::expf(-gxySq / (2.0f * var));
  }

  // TOGXY (paper Fig 11): given the desired scattered direction ω_s (and the
  // outgoing ω_o = -d), the along-ray derivative g_z and the mean gradient ∇μ,
  // solve for the noise gradient g = (g_xy, g_z) whose normalized sum with ∇μ
  // equals the mirror half-vector n = normalize(ω_o + ω_s).
  //
  //   ∇μ + g = λ·n  ⇒  λ = (g_z + ∇μ·d)/(n·d),  g = λ·n − ∇μ.
  //
  // Returns false when the half-vector is degenerate (n·d ≈ 0) or λ ≤ 0.
  MPHD MPINL bool toGxy(const vec3f& wo, const vec3f& ws, const vec3f& d,
                        f32 gz, const vec3f& gradMean,
                        vec3f& g_out, vec3f& n_out, f32& lambda_out) {
    const vec3f n = mps::normalize(wo + ws);
    const f32 nd = mps::dot(n, d);
    if (mps::fabsf(nd) < 1e-6f) return false;
    const f32 lambda = (gz + mps::dot(gradMean, d)) / nd;
    if (lambda <= 0.0f) return false;
    g_out = n * lambda - gradMean;
    n_out = n;
    lambda_out = lambda;
    return true;
  }

  // TOGXYMEASURE (Algorithm 4 line 28): convert a solid-angle pdf p^ω(ω_s) to the
  // gradient-measure pdf of g_xy.
  //   p^n(n) = 4 (n·ω_s) p^ω(ω_s);   return p^n(n) |n·d| / |∇μ+g|².
  MPHD MPINL f32 toGxyMeasure(f32 pOmega, const vec3f& n, const vec3f& ws,
                              const vec3f& d, f32 lambda) {
    const f32 pn = 4.0f * mps::dot(n, ws) * pOmega;
    return pn * mps::fabsf(mps::dot(n, d)) / (lambda * lambda);
  }

}
