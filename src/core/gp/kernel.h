#pragma once

#include "core/math/math_util.h"

namespace mps::core {

  // Squared Exponential sparse-convolution kernel (POD, device-friendly).
  // h(s,p) = exp(-‖p-s‖² / L²),  L² = lengthScale·lengthScale  (scalar, same as
  // the original implementation's squaredNorm()).
  struct SEKernel {
    // Compile-time flag: this kernel is isotropic (a single scalar length scale
    // L² = lengthScale·lengthScale). By subdomain invariance (gpis-1.md「子域不变性」)
    // the GP restricted to a ray is a valid 1D GP with the same covariance, so the
    // marching may replace the 3D noise (27 cells) with a 1D noise (3 cells).
    // An anisotropic kernel ψ_ani(p) = ψ_iso(Mp) would set this false and always
    // fall back to 3D noise (the 1D branch compiles out via if constexpr).
    static constexpr bool isotropic = true;

    f32   sigma;
    vec3f lengthScale;
    f32   kernelRadius;

    MPHD f32 h(const vec3f& s, const vec3f& p) const {
      f32 L2 = mps::dot(lengthScale, lengthScale);
      vec3f d = p - s;
      return mps::expf(-mps::dot(d, d) / L2);
    }

    MPHD vec3f hGrad(const vec3f& C, const vec3f& p) const {
      f32 L2 = mps::dot(lengthScale, lengthScale);
      return h(C, p) * (-2.0f / L2) * (p - C);
    }

    // H_h · v = h(r) * ((-2/L²)·v + (4/L⁴)·(r·v)·r),  r = p - C
    MPHD vec3f hHessianVec(const vec3f& C, const vec3f& p, const vec3f& v) const {
      f32 L2 = mps::dot(lengthScale, lengthScale);
      vec3f r = p - C;
      f32 hv = h(C, p);
      return hv * ((-2.0f / L2) * v + (4.0f / (L2 * L2)) * mps::dot(r, v) * r);
    }

    MPHD f32 oneOverSecondDerivative() const {
      return -mps::dot(lengthScale, lengthScale) / 2.0f;
    }

    MPHD f32 var(f32 impulseDensity) const {
      return (impulseDensity / mps::powf(kernelRadius, 3.0f)) * mps::powf(mps::pi, 1.5f);
    }
  };

}
