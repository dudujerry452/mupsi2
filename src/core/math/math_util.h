#pragma once 

#include "core/basic/types.h"

#include <cmath> 

namespace mps {

  MPHD MPINL f32 rsqrtf(const f32 x) {
    #if defined(__CUDA_ARCH__)
      return rsqrtf(x); 
    #else 
      return 1.0f / std::sqrt(x); 
    #endif 
  }; 
  MPHD MPINL f32 sqrtf(const f32 x) {
    #if defined(__CUDA_ARCH__)
      return sqrtf(x); 
    #else 
      return std::sqrt(x); 
    #endif 
  }
  MPHD MPINL f32 square(const f32 x) {
    return x*x; 
  }

  MPHD MPINL vec3f operator+(const vec3f& a, const vec3f& b) {return {a.x+b.x, a.y+b.y, a.z+b.z}; }
  MPHD MPINL vec3f operator-(const vec3f& a, const vec3f& b) {return {a.x-b.x, a.y-b.y, a.z-b.z}; }
  MPHD MPINL vec3f operator*(const vec3f& a, const vec3f& b) {return {a.x*b.x, a.y*b.y, a.z*b.z}; }
  
  MPHD MPINL vec3f operator-(const vec3f& a) { return {-a.x, -a.y, -a.z}; }
  MPHD MPINL vec3f operator*(const vec3f& a, const f32& b) {return {a.x*b, a.y*b, a.z*b}; }
  MPHD MPINL vec3f operator*(const f32& a, const vec3f& b) { return b*a; }

  MPHD MPINL f32 dot(const vec3f& l, const vec3f& r) {return l.x*r.x+l.y*r.y+l.z*r.z; }
  MPHD MPINL vec3f cross(const vec3f& l, const vec3f& r) {return {l.y*r.z-l.z*r.y, l.z*r.x-l.x*r.z, l.x*r.y-l.y*r.x};  }
  MPHD MPINL f32 norm2(const vec3f& a) {return a.x*a.x + a.y*a.y + a.z*a.z; }
  MPHD MPINL f32 norm(const vec3f& a) { return mps::sqrtf(mps::norm2(a)); }
  MPHD MPINL vec3f normalize(const vec3f& a) {
    f32 invnorm = mps::rsqrtf(mps::norm2(a)); 
    return {a.x*invnorm, a.y*invnorm, a.z*invnorm}; 
  }

  MPHD MPINL vec4f operator+(const vec4f& a, const vec4f& b) {return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; }
  MPHD MPINL vec4f operator-(const vec4f& a, const vec4f& b) {return {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w}; }
  MPHD MPINL vec4f operator*(const vec4f& a, const vec4f& b) {return {a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w}; }
  MPHD MPINL f32 norm2(const vec4f& a) {return a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w; }
  MPHD MPINL f32 norm(const vec4f& a) {return mps::sqrtf(mps::norm2(a)); }
  MPHD MPINL vec4f normalize(const vec4f& a) {
    f32 invnorm = mps::rsqrtf(mps::norm2(a)); 
    return {a.x*invnorm, a.y*invnorm, a.z*invnorm, a.w*invnorm}; 
  }
  MPHD MPINL vec4f quat_normalize(const vec4f& a) {
    f32 invnorm2 = 1.0f / mps::norm2(a); 
    return {a.x*invnorm2, a.y*invnorm2, a.z*invnorm2, a.w*invnorm2}; 
  }
  MPHD MPINL vec4f quat_mul(const vec4f& l, const vec4f& r) {
    return {
      l.w*r.w - l.x*r.x - l.y*r.y - l.z*r.z, 
      l.w*r.x + l.x*r.w + l.y*r.z - l.z*r.y, 
      l.w*r.y + l.y*r.w + l.z*r.x - l.x*r.z, 
      l.w*r.z + l.z*r.w + l.x*r.y - l.y*r.x  
    }; 
  }
  MPHD MPINL vec4f quat_conj(const vec4f& a) {return {-a.x, -a.y, -a.z, a.w}; }




}