#pragma once

#include "core/primitive/primitive.h"
#include "core/math/math_util.h"

namespace mps::core {

  struct Ray {
    vec3f o; 
    vec3f dir;  // must be normalized
    f32 min, max; 

    MPHD MPINL vec3f at(f32 t) const { return o+t*dir; }
  }; 

  struct HitRecord {
    bool hit; 

    f32 t;  // fly distance
    vec3f p; // world hit point 
    vec3f wo; // where it from (in calculation)

    // TBN 
    vec3f ng; // geometry normal 
    vec3f tan;  // tangent 
    vec3f bitan; 

    vec2f uv; 
    u16 bsdfID; 
    byte hasEmis; 
    u16 emisTexID; 
  }; 

  MPHD HitRecord intersect(const Ray& ray, const Sphere& primitive); 
  MPHD bool occulude(const Ray& ray, const Sphere& primitive); 
  // p: where light come from; y: point on light; ng: normal on y
  MPHD f32 surface_pdf(vec3f p, vec3f y, vec3f ng, const Sphere& primitive); 

}