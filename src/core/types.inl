#pragma once 

#include <cstdint> 

namespace mps {
  using u32 = uint32_t; 
  using i32 = int32_t;
  using f32 = float; 

  struct vec3f {f32 x,y,z; }; 
  MPHD inline vec3f operator+(const vec3f& a, const vec3f& b) {return {a.x+b.x, a.y+b.y, a.z+b.z}; }
  MPHD inline vec3f operator*(const vec3f& a, const vec3f& b) {return {a.x*b.x, a.y*b.y, a.z*b.z}; }
  MPHD inline vec3f operator&(const vec3f& a, const vec3f& b) {return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; } // cross 

}