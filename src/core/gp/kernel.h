#pragma once

#include "core/math/math_util.h"

namespace mps::core {

  // Squared Exponential sparse-convolution kernel (POD, device-friendly).
  //
  // Anisotropic form (paper Eq. 25, ψ_ani(p) = ψ_iso(Mp)): the length scale is
  // per-axis L = (Lx, Ly, Lz), M = diag(1/Lx, 1/Ly, 1/Lz), so
  //
  //   h(s,p) = exp(-( (px-sx)²/Lx² + (py-sy)²/Ly² + (pz-sz)²/Lz² )).
  //
  // When Lx == Ly == Lz the kernel is isotropic and reduces to the classic
  // exp(-‖p-s‖²/L²). Subdomain invariance (gpis-1.md「子域不变性」) then lets the
  // marching replace the 3D noise (27 cells) with a 1D noise (3 cells) and enables
  // the §6 analytical gradient NDF — both are gated at runtime on isIsotropic().
  struct SEKernel {
    // Compile-time capability flag: this kernel TYPE is isotropic-capable, so the
    // 1D-noise and analytical-NEE code paths are compiled in. Whether they are
    // actually USED for a given Medium is decided at runtime by isIsotropic().
    static constexpr bool isotropic = true;

    f32   sigma;
    vec3f lengthScale;   // per-axis (Lx, Ly, Lz); equal components => isotropic
    f32   kernelRadius;

    MPHD bool isIsotropic() const {
      return lengthScale.x == lengthScale.y && lengthScale.y == lengthScale.z;
    }

    // Anisotropic squared distance q = Σ (r_i / L_i)².
    MPHD f32 sqDist(const vec3f& s, const vec3f& p) const {
      const vec3f r = p - s;
      return (r.x * r.x) / (lengthScale.x * lengthScale.x) +
             (r.y * r.y) / (lengthScale.y * lengthScale.y) +
             (r.z * r.z) / (lengthScale.z * lengthScale.z);
    }

    MPHD f32 h(const vec3f& s, const vec3f& p) const {
      return mps::expf(-sqDist(s, p));
    }

    // ∇_p h(C,p) = h · (-2) · (r_x/Lx², r_y/Ly², r_z/Lz²),  r = p - C.
    MPHD vec3f hGrad(const vec3f& C, const vec3f& p) const {
      const vec3f r = p - C;
      return h(C, p) * vec3f{-2.0f * r.x / (lengthScale.x * lengthScale.x),
                             -2.0f * r.y / (lengthScale.y * lengthScale.y),
                             -2.0f * r.z / (lengthScale.z * lengthScale.z)};
    }

    // H_h(C,p) · v = h · [ -2 (v_x/Lx², v_y/Ly², v_z/Lz²) + 4 g (g·v) ],
    // with g = (r_x/Lx², r_y/Ly², r_z/Lz²), r = p - C.
    MPHD vec3f hHessianVec(const vec3f& C, const vec3f& p, const vec3f& v) const {
      const vec3f r = p - C;
      const vec3f g{r.x / (lengthScale.x * lengthScale.x),
                    r.y / (lengthScale.y * lengthScale.y),
                    r.z / (lengthScale.z * lengthScale.z)};
      const vec3f m{-2.0f * v.x / (lengthScale.x * lengthScale.x),
                    -2.0f * v.y / (lengthScale.y * lengthScale.y),
                    -2.0f * v.z / (lengthScale.z * lengthScale.z)};
      return h(C, p) * (m + 4.0f * g * mps::dot(g, v));
    }

    // Solve β = H_h(C,C)⁻¹ · target. At r = 0, h = 1 and
    // H_h(C,C) = -2·diag(1/Lx², 1/Ly², 1/Lz²), so
    // β = -(1/2)·diag(Lx², Ly², Lz²)·target.
    MPHD vec3f solveBeta(const vec3f& target) const {
      return {-0.5f * lengthScale.x * lengthScale.x * target.x,
              -0.5f * lengthScale.y * lengthScale.y * target.y,
              -0.5f * lengthScale.z * lengthScale.z * target.z};
    }

    // ∫ h² dx = (π/2)^{3/2} · Lx·Ly·Lz (anisotropic SE normalization factor).
    MPHD f32 integralH2() const {
      return mps::powf(0.5f * mps::pi, 1.5f) * lengthScale.x * lengthScale.y * lengthScale.z;
    }
  };

}
