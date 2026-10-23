#pragma once

#include "core/basic/types.h"
#include "core/bvh/bvh.h"
#include "core/math/math_util.h"

namespace mps::core {

  // Isotropic heat-kernel texture (HKTEx, Foti et al. 2026) — the "geodesic
  // Gaussians" defined on the mesh surface via Laplace-Beltrami eigenfunctions.
  //
  //   h_t(p, s) = Σ_k e^{-t λ_k} φ_k(p) φ_k(s)            (paper Eq. 3, m* = 1)
  //   h̄ = h / h_self,  h_self = Σ_k e^{-t λ_k} φ_k(s)²     (normalize to 1 at s)
  //   α_s(p) = rescaled soft-step of h̄                       (paper Eq. 6)
  //   C(p)   = c_mean + Σ_s α_s c_s / max(Σ_s α_s, 1)        (paper Eq. 7)
  //
  // A 3D query point is mapped to the surface by CLOSEST-POINT projection onto the
  // mesh (face + barycentric), then φ_k is barycentrically interpolated there.
  // POD / device-friendly; pointers rebound by the backend after upload.
  struct HeatKernel {
    // Mesh geometry for the closest-point projection.
    const vec3f*    vertices      = nullptr;
    const u32*      faces         = nullptr;   // 3×u32 per triangle
    const BVHNode*  bvhNodes      = nullptr;
    const u32*      bvhTriIndices = nullptr;
    u32 bvhRoot = 0, bvhNodeOffset = 0, bvhTriOffset = 0;

    // Spectral data.
    const f32*     phi         = nullptr;   // N×K eigenvectors (vertex-major)
    const f32*     lam         = nullptr;   // K eigenvalues (ascending, λ0≈0)
    const f32*     kernelPhi   = nullptr;   // S×K source eigenvectors φ(s)
    const RGB_F32* kernelColor = nullptr;   // S source colors
    const f32*     kernelTau   = nullptr;   // S thresholds τ
    const f32*     kernelSharp = nullptr;   // S sharpnesses ς
    u32 N = 0, K = 0, S = 0;
    f32 t = 1e-2f;
    f32 bhSigma = 0.1f;   // biharmonic-distance Gaussian sigma (paper Eq. 5)
    RGB_F32 meanColor{0.5f, 0.5f, 0.5f};

    MPHD bool empty() const {
      return N == 0 || K == 0 || phi == nullptr || lam == nullptr || vertices == nullptr;
    }

    MPHD static MPINL f32 sigmoid(f32 v) { return 1.0f / (1.0f + mps::expf(-v)); }

    // Rescaled soft-step: 0 at x=0, 1 at x=1 (paper Eq. 6).
    MPHD static MPINL f32 rescaledSoftStep(f32 x, f32 tau, f32 sharp) {
      const f32 y  = sigmoid(sharp * (x - tau));
      const f32 y0 = sigmoid(sharp * (0.0f - tau));
      const f32 y1 = sigmoid(sharp * (1.0f - tau));
      return mps::clampf((y - y0) / (y1 - y0 + 1e-8f), 0.0f, 1.0f);
    }

    MPHD RGB_F32 evalColor(const vec3f& p) const {
      // Closest point on the mesh → face + barycentric.
      u32 tri;
      vec3f cp;
      closestPointBVH(p, bvhNodes + bvhNodeOffset, bvhTriIndices + bvhTriOffset,
                      vertices, faces, bvhRoot, tri, cp);

      const u32 i0 = faces[3 * tri + 0], i1 = faces[3 * tri + 1], i2 = faces[3 * tri + 2];
      const vec3f v0 = vertices[i0], v1 = vertices[i1], v2 = vertices[i2];
      const vec3f e0 = v1 - v0, e1 = v2 - v0, e2 = cp - v0;
      const f32 d00 = mps::dot(e0, e0), d01 = mps::dot(e0, e1), d11 = mps::dot(e1, e1);
      const f32 d20 = mps::dot(e2, e0), d21 = mps::dot(e2, e1);
      const f32 denom = d00 * d11 - d01 * d01;
      f32 u, v, w;
      if (mps::fabsf(denom) > 1e-12f) {
        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = 1.0f - v - w;
      } else { u = 1.0f; v = 0.0f; w = 0.0f; }
      u = mps::maxf(u, 0.0f); v = mps::maxf(v, 0.0f); w = mps::maxf(w, 0.0f);
      const f32 ws = u + v + w;
      if (ws > 1e-12f) { u /= ws; v /= ws; w /= ws; }

      // Decay factor exp(-t λ_k), reused for every kernel.
      // φ_k(p) via barycentric interpolation of the per-vertex eigenvectors.
      RGB_F32 acc{0.0f, 0.0f, 0.0f};
      f32 accW = 0.0f;
      for (u32 s = 0; s < S; ++s) {
        f32 h = 0.0f, hSelf = 0.0f, dBH2 = 0.0f;
        for (u32 k = 0; k < K; ++k) {
          const f32 decay = mps::expf(-t * lam[k]);
          const f32 phiP = u * phi[size_t(i0) * K + k] + v * phi[size_t(i1) * K + k] +
                           w * phi[size_t(i2) * K + k];
          const f32 phiS = kernelPhi[size_t(s) * K + k];
          h      += decay * phiP * phiS;
          hSelf  += decay * phiS * phiS;
          if (k > 0 && lam[k] > 1e-12f) {   // skip the constant mode (λ0=0)
            const f32 d = phiP - phiS;
            dBH2 += (d * d) / (lam[k] * lam[k]);
          }
        }
        f32 hBar = h / (hSelf + 1e-8f);
        hBar *= mps::expf(-dBH2 / (2.0f * bhSigma * bhSigma));   // biharmonic localization (Eq. 5)
        const f32 alpha = rescaledSoftStep(hBar, kernelTau[s], kernelSharp[s]);
        acc   = acc + kernelColor[s] * alpha;
        accW += alpha;
      }
      // Weighted average of the kernel residual colors. Normalizing by accW
      // (instead of max(accW, 1)) keeps the color fully saturated even in
      // sparsely-covered regions — chosen to make the heat-kernel scheme as
      // vivid as the other three color schemes for the experiments.
      const f32 inv = (accW > 1e-3f) ? (1.0f / accW) : 0.0f;
      RGB_F32 c = meanColor + acc * inv;

      // Saturation boost: overlapping heat kernels average toward gray; push the
      // chroma back out so the heat-kernel scheme reads as vividly as the other
      // three color schemes (experiment goal: maximally prominent colors).
      const f32 lum = 0.3333333f * (c.r + c.g + c.b);
      c = {lum + 2.6f * (c.r - lum), lum + 2.6f * (c.g - lum), lum + 2.6f * (c.b - lum)};
      return c;
    }
  };

}
