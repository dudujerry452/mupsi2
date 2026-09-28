#include "core/tracer/intersect.h"
#include "core/math/math_util.h"

namespace mps::core {


  MPHD HitRecord Intersect<PrimitiveType::Sphere>::run(
    const Ray& ray, 
    const Primitive<PrimitiveType::Sphere>& primitive
  ) {

    HitRecord hr; 
    hr.hit = false;

    const auto C = primitive.info.wpos; 
    const auto O = ray.o; 
    const auto D = ray.dir; // normalized 
    const auto V = C - O; 
    const auto R = primitive.radius; 

    f32 disc = mps::sqrtf(
      mps::square(mps::dot(D, V)) - 
      (mps::dot(V,V) - mps::square(R))
    );
    
    if(disc < 0.0f) return hr; 

    f32 k = (mps::dot(D, V)); 
    f32 t1 = -k + disc, t2 = -k - disc; 
    f32 t = (t1 >= 0.0f) ? t1 : t2;  
    vec3f p = O + D*t; 
    vec3f ng = mps::normalize(p-C); 
    ng = (t1 >= 0.0f) ? ng : -ng; 

    hr.p = p; 
    hr.ng = ng; 
    hr.t = t; 
    hr.wo = -ray.dir; 
    return hr; 

  }; 
}