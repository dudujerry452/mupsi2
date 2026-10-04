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
    const auto O = ray.o + ray.at(ray.min); 
    const auto D = ray.dir; // normalized 
    const auto V = O - C; 
    const auto R = primitive.radius; 

    // solve 2 var equation
    f32 k = mps::dot(D, V); 

    f32 disc = 
      mps::square(k) - 
      (mps::dot(V,V) - mps::square(R))
    ;

    if(disc < 0.0f) return hr; 

    disc = mps::sqrtf(disc); 
    f32 t1 = -k - disc, t2 = -k + disc; 

    if(t2 < 0.0f) return hr; 
    
    f32 t = (t1 >= 0.0f) ? t1 : t2;  if(t > ray.max) return hr; 
    vec3f p = O + D*t; 
    vec3f ng = mps::normalize(p-C); 
    ng = (t1 >= 0.0f) ? ng : -ng; 

    // TBN 
    vec3f local_ng = mps::quat_rotate_inv(primitive.info.quater, ng); 
    vec3f T = (mps::fabsf(local_ng.y) > 0.999f) ? vec3f{1.0f, 0.0f, 0.0f} : vec3f{-local_ng.z, 0, local_ng.x}; 
    T = mps::normalize(T); 
    vec3f B = mps::cross(local_ng, T); 
    B = (mps::dot(B, {0, -1, 0}) < 0.0f) ? -B : B; 

    // back to world coordinate
    T = mps::quat_rotate(primitive.info.quater, T);
    B = mps::quat_rotate(primitive.info.quater, B); 

    f32 u = (mps::atan2f(local_ng.z, local_ng.x) + mps::pi) * mps::inv_two_pi; 
    f32 v = mps::acosf(local_ng.y) * mps::inv_pi; 

    hr.hit = true; 
    hr.p = p; 
    hr.ng = ng; 
    hr.t = t; 
    hr.wo = -ray.dir; 
    hr.uv = {u, v}; 
    hr.tan = T; 
    hr.bitan = B; 
  
    return hr; 

  }; 
}