#include "core/types.h"

namespace mps {
  MPHD vec3f operator+(const vec3f& a, const vec3f& b) {return {a.x+b.x, a.y+b.y, a.z+b.z}; }
  MPHD vec3f operator*(const vec3f& a, const vec3f& b) {return {a.x*b.x, a.y*b.y, a.z*b.z}; }
  MPHD vec3f operator&(const vec3f& a, const vec3f& b) {return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; } // cross
}
