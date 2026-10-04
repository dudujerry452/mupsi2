#pragma once

#include "core/primitive/primitive.h"

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
    u16 texId; 
    u16 bsdfId; 
    u16 lightId;  
  }; 

  template<PrimitiveType T> 
  struct Intersect; 

  template<> struct Intersect<PrimitiveType::Sphere> {
    MPHD static HitRecord run(const Ray& ray, const Primitive<PrimitiveType::Sphere>& primitive); 
  }; 




}