#pragma once

#include "core/gp/kernel.h"

namespace mps::core {

  // Sparse convolution noise (gpis-1.md「离散卷积」): ψ(p) = Σᵢ wᵢ·h(sᵢ,p).
  //
  // Impulses sᵢ are scattered on a regular grid (cell size = kernelRadius) with a
  // fixed pointsPerCell impulses per cell, each at a pseudo-random position with a
  // ±1 (Bernoulli) weight. Density λ = pointsPerCell / cellSize³. The raw sum is
  // normalized by √(λ·(πL²/2)^{3/2}) so that ψ has covariance
  // σ²·exp(-‖p-q‖²/(2L²)), independent of L and λ.
  struct SparseConvNoise {
    SEKernel kernel;
    u32 pointsPerCell = 3;
    f32 cellSize = 1.0f;   // = kernelRadius

    // λ · (πL²/2)^{3/2}, the variance of the raw field.
    MPHD f32 variance() const {
      f32 L2 = mps::dot(kernel.lengthScale, kernel.lengthScale);
      f32 lambda = f32(pointsPerCell) / (cellSize * cellSize * cellSize);
      return lambda * mps::powf(0.5f * mps::pi * L2, 1.5f);
    }

    // λ₁ · (πL²/2)^{1/2}, the variance of the raw 1D field along a ray.
    MPHD f32 variance1D() const {
      f32 L2 = mps::dot(kernel.lengthScale, kernel.lengthScale);
      f32 lambda = f32(pointsPerCell) / cellSize;
      return lambda * mps::sqrtf(0.5f * mps::pi * L2);
    }

    // Un-normalized Σ wᵢ h(sᵢ,p) and its gradient (vec4f{ψ, ∇ψ}).
    MPHD vec4f evalRaw(const vec3f& p, u32 seed) const;

    // Normalized noise: vec4f{ψ, ∇ψ} with unit-shape covariance, amplitude σ.
    MPHD vec4f eval(const vec3f& p, u32 seed) const {
      f32 scale = kernel.sigma / mps::sqrtf(variance());
      return evalRaw(p, seed) * scale;
    }

    // 1D noise along a ray (isotropic kernels only): vec2f{ψ, ∂ψ/∂t} at distance
    // t along the ray. Same normalization as eval(), so ψ has covariance
    // exp(-r²/(2L²)) restricted to the ray (subdomain invariance).
    MPHD vec2f eval1D(f32 t, u32 seed) const;
  };

}
