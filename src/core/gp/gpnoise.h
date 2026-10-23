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

    // Local σ at p (non-stationary field if present, else the constant kernel).
    MPHD f32 sigmaAt(const vec3f& p) const {
      if (field && !field->empty()) return field->eval(p).x;
      return kernel.sigma;
    }

    // Per-axis squared length scale {Lx², Ly², Lz²} at p. A non-stationary field
    // (isotropic, scalar L) overrides all three axes with the local L²; otherwise
    // the kernel's per-axis lengthScale is used (anisotropic when components differ).
    MPHD vec3f lengthScale2At(const vec3f& p) const {
      vec3f L = kernel.lengthScale;
      if (field && !field->empty()) {
        const f32 l = field->eval(p).y;
        L = {l, l, l};
      }
      return {L.x * L.x, L.y * L.y, L.z * L.z};
    }

    // λ · (π/2)^{3/2} · Lx·Ly·Lz, the variance of the raw field (anisotropic SE).
    MPHD f32 variance(const vec3f& p) const {
      const vec3f L2 = lengthScale2At(p);
      const f32 Lx = mps::sqrtf(L2.x), Ly = mps::sqrtf(L2.y), Lz = mps::sqrtf(L2.z);
      const f32 lambda = f32(pointsPerCell) / (cellSize * cellSize * cellSize);
      return lambda * mps::powf(0.5f * mps::pi, 1.5f) * Lx * Ly * Lz;
    }

    // λ₁ · (πL²/2)^{1/2}, the variance of the raw 1D field along a ray (local L;
    // only ever called for isotropic kernels).
    MPHD f32 variance1D(const vec3f& p) const {
      const f32 L = mps::sqrtf(lengthScale2At(p).x);
      const f32 lambda = f32(pointsPerCell) / cellSize;
      return lambda * mps::sqrtf(0.5f * mps::pi * L * L);
    }

    // Un-normalized Σ wᵢ h(sᵢ,p) and its gradient (vec4f{ψ, ∇ψ}), using the
    // local length scale L(p) for the kernel support/decay.
    MPHD vec4f evalRaw(const vec3f& p, u32 seed) const;

    // Normalized noise: vec4f{ψ, ∇ψ} with unit-shape covariance, amplitude σ(p).
    MPHD vec4f eval(const vec3f& p, u32 seed) const {
      const f32 scale = sigmaAt(p) / mps::sqrtf(variance(p));
      return evalRaw(p, seed) * scale;
    }

    // 1D noise along a ray (isotropic kernels only): vec2f{ψ, ∂ψ/∂t} at world
    // point p = o + t·d. Same normalization as eval().
    MPHD vec2f eval1D(const vec3f& p, f32 t, u32 seed) const;
  };

}
