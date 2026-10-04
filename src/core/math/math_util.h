#pragma once 

#include "core/basic/types.h"

#include <cmath> 

namespace mps {

  constexpr f32 pi           = 3.14159265358979323846f;
  constexpr f32 two_pi       = 6.28318530717958647692f;
  constexpr f32 half_pi      = 1.57079632679489661923f;

  constexpr f32 inv_pi       = 0.31830988618379067154f; 
  constexpr f32 inv_two_pi   = 0.15915494309189533577f; 


  MPHD MPINL f32 rsqrtf(f32 x) {
    #if defined(__CUDA_ARCH__)
      return ::rsqrtf(x); 
    #else 
      return 1.0f / std::sqrt(x); 
    #endif 
  }; 
  MPHD MPINL f32 sqrtf(f32 x) {
    #if defined(__CUDA_ARCH__)
      return ::sqrtf(x); 
    #else 
      return std::sqrt(x); 
    #endif 
  }
  MPHD MPINL f32 square(f32 x) {
    return x*x; 
  }
  
  MPHD MPINL f32 cosf(f32 x) {
    #if defined(__CUDA_ARCH__) 
      return ::cosf(x); 
    #else 
      return std::cosf(x); 
    #endif
  }

  MPHD MPINL f32 sinf(f32 x) {
    #if defined(__CUDA_ARCH__) 
      return ::sinf(x); 
    #else 
      return std::sinf(x); 
    #endif
  }
  
  MPHD MPINL f32 atan2f(f32 y, f32 x) {
    #if defined(__CUDA_ARCH__) 
      return ::atan2f(y, x); 
    #else 
      return std::atan2(y, x); 
    #endif
  }

  MPHD MPINL f32 acosf(f32 x) {
        #if defined(__CUDA_ARCH__) 
      return ::acosf(x); 
    #else 
      return std::acos(x); 
    #endif
  }

  MPHD MPINL f32 asinf(f32 x) {
        #if defined(__CUDA_ARCH__) 
      return ::asinf(x); 
    #else 
      return std::asin(x); 
    #endif
  }

  MPHD MPINL f32 fabsf(f32 x) {
  #ifdef __CUDA_ARCH__
      return ::fabsf(x);
  #else
      return std::fabs(x); 
  #endif
  }

  MPHD MPINL vec3f operator+(vec3f a, vec3f b) {return {a.x+b.x, a.y+b.y, a.z+b.z}; }
  MPHD MPINL vec3f operator-(vec3f a, vec3f b) {return {a.x-b.x, a.y-b.y, a.z-b.z}; }
  MPHD MPINL vec3f operator*(vec3f a, vec3f b) {return {a.x*b.x, a.y*b.y, a.z*b.z}; }
  
  MPHD MPINL vec3f operator-(vec3f a) { return {-a.x, -a.y, -a.z}; }
  MPHD MPINL vec3f operator*(vec3f a, f32 b) {return {a.x*b, a.y*b, a.z*b}; }
  MPHD MPINL vec3f operator*(f32 a, vec3f b) { return b*a; }

  MPHD MPINL f32 dot(vec3f l, vec3f r) {return l.x*r.x+l.y*r.y+l.z*r.z; }
  MPHD MPINL vec3f cross(vec3f l, vec3f r) {return {l.y*r.z-l.z*r.y, l.z*r.x-l.x*r.z, l.x*r.y-l.y*r.x};  }
  MPHD MPINL f32 norm2(vec3f a) {return a.x*a.x + a.y*a.y + a.z*a.z; }
  MPHD MPINL f32 norm(vec3f a) { return mps::sqrtf(mps::norm2(a)); }
  MPHD MPINL vec3f normalize(vec3f a) {
    f32 invnorm = mps::rsqrtf(mps::norm2(a)); 
    return {a.x*invnorm, a.y*invnorm, a.z*invnorm}; 
  }

  MPHD MPINL vec4f operator+(vec4f a, vec4f b) {return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; }
  MPHD MPINL vec4f operator-(vec4f a, vec4f b) {return {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w}; }
  MPHD MPINL vec4f operator*(vec4f a, vec4f b) {return {a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w}; }
  MPHD MPINL f32 norm2(vec4f a) {return a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w; }
  MPHD MPINL f32 norm(vec4f a) {return mps::sqrtf(mps::norm2(a)); }
  MPHD MPINL vec4f normalize(vec4f a) {
    f32 invnorm = mps::rsqrtf(mps::norm2(a)); 
    return {a.x*invnorm, a.y*invnorm, a.z*invnorm, a.w*invnorm}; 
  }

  MPHD MPINL vec4f quat_init(vec3f axis, f32 theta) { f32 ht = theta*0.5f; return { axis.x*ht, axis.y*ht, axis.z*ht,  mps::cosf(ht)}; }
  MPHD MPINL vec3f quat_rotate(vec4f q, vec3f v) {return {v + mps::cross(2.0f*q.u, mps::cross(q.u, v) + q.w*v)}; }
  MPHD MPINL vec3f quat_rotate_inv(vec4f q, vec3f v) {return {v - mps::cross(2.0f*q.u, -mps::cross(q.u, v) + q.w*v)}; }
  MPHD MPINL vec4f quat_normalize(vec4f a) {
    f32 invnorm2 = 1.0f / mps::norm2(a); 
    return {a.x*invnorm2, a.y*invnorm2, a.z*invnorm2, a.w*invnorm2}; 
  }
  MPHD MPINL vec4f quat_mul(vec4f l, vec4f r) {
    return {
      l.w*r.x + l.x*r.w + l.y*r.z - l.z*r.y, 
      l.w*r.y + l.y*r.w + l.z*r.x - l.x*r.z, 
      l.w*r.z + l.z*r.w + l.x*r.y - l.y*r.x, 
      l.w*r.w - l.x*r.x - l.y*r.y - l.z*r.z  
    }; 
  }
  MPHD MPINL vec4f quat_conj(vec4f a) {return {-a.x, -a.y, -a.z, a.w}; }




}
