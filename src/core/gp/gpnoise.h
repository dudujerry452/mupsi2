#pragma once

#include "core/gp/kernel.h"
#include "core/gp/kernelfield.h"

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
    const KernelField* field = nullptr;   // optional non-stationary {σ, L}

    // Local {σ, L²} at p (non-stationary field if present, else the constant kernel).
    MPHD vec2f sigmaL2At(const vec3f& p) const {
      if (field && !field->empty()) {
        const vec2f sl = field->eval(p);
        return {sl.x, sl.y * sl.y};
      }
      return {kernel.sigma, mps::dot(kernel.lengthScale, kernel.lengthScale)};
    }

    // λ · (πL²/2)^{3/2}, the variance of the raw field (local L).
    MPHD f32 variance(const vec3f& p) const {
      const f32 L2 = sigmaL2At(p).y;
      const f32 lambda = f32(pointsPerCell) / (cellSize * cellSize * cellSize);
      return lambda * mps::powf(0.5f * mps::pi * L2, 1.5f);
    }

    // λ₁ · (πL²/2)^{1/2}, the variance of the raw 1D field along a ray (local L).
    MPHD f32 variance1D(const vec3f& p) const {
      const f32 L2 = sigmaL2At(p).y;
      const f32 lambda = f32(pointsPerCell) / cellSize;
      return lambda * mps::sqrtf(0.5f * mps::pi * L2);
    }

    // Un-normalized Σ wᵢ h(sᵢ,p) and its gradient (vec4f{ψ, ∇ψ}), using the
    // local length scale L(p) for the kernel support/decay.
    MPHD vec4f evalRaw(const vec3f& p, u32 seed) const;

    // Normalized noise: vec4f{ψ, ∇ψ} with unit-shape covariance, amplitude σ(p).
    MPHD vec4f eval(const vec3f& p, u32 seed) const {
      const f32 scale = sigmaL2At(p).x / mps::sqrtf(variance(p));
      return evalRaw(p, seed) * scale;
    }

    // 1D noise along a ray (isotropic kernels only): vec2f{ψ, ∂ψ/∂t} at world
    // point p = o + t·d. Same normalization as eval().
    MPHD vec2f eval1D(const vec3f& p, f32 t, u32 seed) const;
  };

}
