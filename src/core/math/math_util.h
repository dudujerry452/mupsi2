#pragma once

#include "core/basic/types.h"

#include <cmath>

namespace mps {

  constexpr f32 pi           = 3.14159265358979323846f;
  constexpr f32 two_pi       = 6.28318530717958647692f;
  constexpr f32 half_pi      = 1.57079632679489661923f;
  constexpr f32 inv_pi       = 0.31830988618379067154f;
  constexpr f32 inv_two_pi   = 0.15915494309189533577f;

  MPHD MPINL f32 rsqrtf(const f32 x) {
    #if defined(__CUDA_ARCH__)
      return ::rsqrtf(x);
    #else
      return 1.0f / std::sqrt(x);
    #endif
  }
  MPHD MPINL f32 sqrtf(const f32 x) {
    #if defined(__CUDA_ARCH__)
      return ::sqrtf(x);
    #else
      return std::sqrt(x);
    #endif
  }
  MPHD MPINL f32 square(const f32 x) { return x*x; }

  MPHD MPINL f32 cosf(const f32 x) {
    #if defined(__CUDA_ARCH__)
      return ::cosf(x);
    #else
      return std::cosf(x);
    #endif
  }
  MPHD MPINL f32 sinf(const f32 x) {
    #if defined(__CUDA_ARCH__)
      return ::sinf(x);
    #else
      return std::sinf(x);
    #endif
  }
  MPHD MPINL f32 tanf(const f32 x) {
    #if defined(__CUDA_ARCH__)
      return ::tanf(x);
    #else
      return std::tanf(x);
    #endif
  }
  MPHD MPINL f32 atan2f(const f32 y, const f32 x) {
    #if defined(__CUDA_ARCH__)
      return ::atan2f(y, x);
    #else
      return std::atan2(y, x);
    #endif
  }
  MPHD MPINL f32 acosf(const f32 x) {
    #if defined(__CUDA_ARCH__)
      return ::acosf(x);
    #else
      return std::acos(x);
    #endif
  }
  MPHD MPINL f32 asinf(const f32 x) {
    #if defined(__CUDA_ARCH__)
      return ::asinf(x);
    #else
      return std::asin(x);
    #endif
  }
  MPHD MPINL f32 fabsf(const f32 x) {
    #ifdef __CUDA_ARCH__
      return ::fabsf(x);
    #else
      return std::fabs(x);
    #endif
  }
  MPHD MPINL f32 powf(const f32 x, const f32 y) {
    #if defined(__CUDA_ARCH__)
      return ::powf(x, y);
    #else
      return std::pow(x, y);
    #endif
  }
  MPHD MPINL f32 expf(const f32 x) {
    #if defined(__CUDA_ARCH__)
      return ::expf(x);
    #else
      return std::expf(x);
    #endif
  }
  MPHD MPINL f32 logf(const f32 x) {
    #if defined(__CUDA_ARCH__)
      return ::logf(x);
    #else
      return std::logf(x);
    #endif
  }
  MPHD MPINL f32 floorf(const f32 x) {
    #if defined(__CUDA_ARCH__)
      return ::floorf(x);
    #else
      return std::floorf(x);
    #endif
  }

  // ——— scalar min/max/clamp/lerp ———
  MPHD MPINL f32 minf(f32 a, f32 b) { return a < b ? a : b; }
  MPHD MPINL f32 maxf(f32 a, f32 b) { return a > b ? a : b; }
  MPHD MPINL f32 clampf(f32 x, f32 lo, f32 hi) { return x < lo ? lo : (x > hi ? hi : x); }
  MPHD MPINL f32 lerpf(f32 a, f32 b, f32 t) { return a + (b - a) * t; }

  // ——— vec2f ———
  MPHD MPINL vec2f operator+(const vec2f a, const vec2f b) { return {a.x+b.x, a.y+b.y}; }
  MPHD MPINL vec2f operator-(const vec2f a, const vec2f b) { return {a.x-b.x, a.y-b.y}; }
  MPHD MPINL vec2f operator*(const vec2f a, const f32 b) { return {a.x*b, a.y*b}; }
  MPHD MPINL vec2f operator*(const f32 a, const vec2f b) { return b*a; }
  MPHD MPINL f32 dot(const vec2f l, const vec2f r) { return l.x*r.x + l.y*r.y; }
  MPHD MPINL f32 norm2(const vec2f a) {return a.x*a.x + a.y*a.y; }
  MPHD MPINL vec2f normalize(const vec2f a) {return a * mps::rsqrtf(norm2(a)); }

  // ——— vec3f ———
  MPHD MPINL vec3f operator+(const vec3f a, const vec3f b) {return {a.x+b.x, a.y+b.y, a.z+b.z}; }
  MPHD MPINL vec3f operator-(const vec3f a, const vec3f b) {return {a.x-b.x, a.y-b.y, a.z-b.z}; }
  MPHD MPINL vec3f operator*(const vec3f a, const vec3f b) {return {a.x*b.x, a.y*b.y, a.z*b.z}; }
  MPHD MPINL vec3f operator/(const vec3f a, const f32 b) { const f32 i = 1.0f/b; return {a.x*i, a.y*i, a.z*i}; }
  MPHD MPINL vec3f operator-(const vec3f a) { return {-a.x, -a.y, -a.z}; }
  MPHD MPINL vec3f operator*(const vec3f a, const f32 b) {return {a.x*b, a.y*b, a.z*b}; }
  MPHD MPINL vec3f operator*(const f32 a, const vec3f b) { return b*a; }

  MPHD MPINL f32 dot(const vec3f l, const vec3f r) {return l.x*r.x+l.y*r.y+l.z*r.z; }
  MPHD MPINL vec3f cross(const vec3f l, const vec3f r) {return {l.y*r.z-l.z*r.y, l.z*r.x-l.x*r.z, l.x*r.y-l.y*r.x};  }
  MPHD MPINL f32 norm2(const vec3f a) {return a.x*a.x + a.y*a.y + a.z*a.z; }
  MPHD MPINL f32 norm(const vec3f a) { return mps::sqrtf(mps::norm2(a)); }
  MPHD MPINL vec3f normalize(const vec3f a) {
    const f32 invnorm = mps::rsqrtf(mps::norm2(a));
    return {a.x*invnorm, a.y*invnorm, a.z*invnorm};
  }
  MPHD MPINL vec3f minv(const vec3f a, const vec3f b) { return {mps::minf(a.x,b.x), mps::minf(a.y,b.y), mps::minf(a.z,b.z)}; }
  MPHD MPINL vec3f maxv(const vec3f a, const vec3f b) { return {mps::maxf(a.x,b.x), mps::maxf(a.y,b.y), mps::maxf(a.z,b.z)}; }
  MPHD MPINL vec3f clampv(const vec3f x, const vec3f lo, const vec3f hi) {
    return {mps::clampf(x.x,lo.x,hi.x), mps::clampf(x.y,lo.y,hi.y), mps::clampf(x.z,lo.z,hi.z)};
  }
  MPHD MPINL vec3f lerpv(const vec3f a, const vec3f b, const f32 t) { return a + (b - a) * t; }

  // ——— vec4f ———
  MPHD MPINL vec4f operator+(const vec4f a, const vec4f b) {return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; }
  MPHD MPINL vec4f operator-(const vec4f a, const vec4f b) {return {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w}; }
  MPHD MPINL vec4f operator*(const vec4f a, const vec4f b) {return {a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w}; }
  MPHD MPINL vec4f operator*(const vec4f a, const f32 b) {return {a.x*b, a.y*b, a.z*b, a.w*b}; }
  MPHD MPINL vec4f operator*(const f32 a, const vec4f b) { return b*a; }
  MPHD MPINL f32 norm2(const vec4f a) {return a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w; }
  MPHD MPINL f32 norm(const vec4f a) {return mps::sqrtf(mps::norm2(a)); }
  MPHD MPINL vec4f normalize(const vec4f a) {
    const f32 invnorm = mps::rsqrtf(mps::norm2(a));
    return {a.x*invnorm, a.y*invnorm, a.z*invnorm, a.w*invnorm};
  }

  // require axis is normalized
  MPHD MPINL vec4f quat_init(const vec3f axis, const f32 theta) { 
    const f32 ht = theta * 0.5f; 
    const f32 hts = mps::sinf(ht); 
    return { axis.x*hts, axis.y*hts, axis.z*hts,  mps::cosf(ht)}; 
  }
  // require q,v is unit normalized
  MPHD MPINL vec3f quat_rotate(const vec4f q, const vec3f v) {return {v + mps::cross(2.0f*q.u, mps::cross(q.u, v) + q.w*v)}; }
  MPHD MPINL vec3f quat_rotate_inv(const vec4f q, const vec3f v) {return {v - mps::cross(2.0f*q.u, -mps::cross(q.u, v) + q.w*v)}; }
  MPHD MPINL vec4f quat_mul(const vec4f l, const vec4f r) {
    return {
      l.w*r.x + l.x*r.w + l.y*r.z - l.z*r.y,
      l.w*r.y + l.y*r.w + l.z*r.x - l.x*r.z,
      l.w*r.z + l.z*r.w + l.x*r.y - l.y*r.x,
      l.w*r.w - l.x*r.x - l.y*r.y - l.z*r.z
    };
  }
  MPHD MPINL vec4f quat_conj(const vec4f a) {return {-a.x, -a.y, -a.z, a.w}; }
  MPHD MPINL vec4f quat_inverse(const vec4f a) {
    vec4f conj = quat_conj(a); 
    const f32 invnorm2 = 1.0f / mps::norm2(a);
    return {conj.x*invnorm2, conj.y*invnorm2, conj.z*invnorm2, conj.w*invnorm2};
  }
  MPHD MPINL vec4f quat_from_matrix(const vec3f right, const vec3f up, const vec3f forward) {
      vec4f quat; 

      f32 m00 = right.x, m01 = up.x, m02 = -forward.x;
      f32 m10 = right.y, m11 = up.y, m12 = -forward.y;
      f32 m20 = right.z, m21 = up.z, m22 = -forward.z;

      f32 trace = m00 + m11 + m22;

      if (trace > 0.0f) {
          f32 s = 2.0f * mps::sqrtf(trace + 1.0f);
          f32 s_inv = 1.0f / s;

          quat = vec4f{
              (m21 - m12) * s_inv,
              (m02 - m20) * s_inv,
              (m10 - m01) * s_inv,
              0.25f * s
          };
      } else if (m00 > m11 && m00 > m22) {
          f32 s = 2.0f * mps::sqrtf(1.0f + m00 - m11 - m22);
          f32 s_inv = 1.0f / s;

          quat = vec4f{
              0.25f * s,
              (m01 + m10) * s_inv,
              (m02 + m20) * s_inv,
              (m21 - m12) * s_inv
          };
      } else if (m11 > m22) {
          f32 s = 2.0f * mps::sqrtf(1.0f + m11 - m00 - m22);
          f32 s_inv = 1.0f / s;

          quat = vec4f{
              (m01 + m10) * s_inv,
              0.25f * s,
              (m12 + m21) * s_inv,
              (m02 - m20) * s_inv
          };
      } else {
          f32 s = 2.0f * mps::sqrtf(1.0f + m22 - m00 - m11);
          f32 s_inv = 1.0f / s;

          quat = vec4f{
              (m02 + m20) * s_inv,
              (m12 + m21) * s_inv,
              0.25f * s,
              (m10 - m01) * s_inv
          };
      }
      return quat; 
  }

  // ——— RGB_F32 ———
  MPHD MPINL RGB_F32& operator+=(RGB_F32& a, const RGB_F32 b) { a.r+=b.r; a.g+=b.g; a.b+=b.b; return a; }
  MPHD MPINL RGB_F32& operator-=(RGB_F32& a, const RGB_F32 b) { a.r-=b.r; a.g-=b.g; a.b-=b.b; return a; }
  MPHD MPINL RGB_F32 operator+(const RGB_F32 a, const RGB_F32 b) { return {a.r+b.r, a.g+b.g, a.b+b.b}; }
  MPHD MPINL RGB_F32 operator-(const RGB_F32 a, const RGB_F32 b) { return {a.r-b.r, a.g-b.g, a.b-b.b}; }
  MPHD MPINL RGB_F32 operator*(const RGB_F32 a, const RGB_F32 b) { return {a.r*b.r, a.g*b.g, a.b*b.b}; }
  MPHD MPINL RGB_F32 operator*(const RGB_F32 a, const f32 b) { return {a.r*b, a.g*b, a.b*b}; }
  MPHD MPINL RGB_F32 operator*(const f32 a, const RGB_F32 b) { return b*a; }
  MPHD MPINL RGB_F32 operator/(const RGB_F32 a, const f32 b) { const f32 i = 1.0f/b; return {a.r*i, a.g*i, a.b*i}; }

}
