#include "core/bsdf/bsdf.h"
#include "core/math/math_util.h"


namespace mps::core {

  MPHD MPINL RGB_F32 BsdfImpl<BsdfType::Lambertain>::f(vec3f wi, vec3f wo, const BsdfParam& par) {
    if (wi.z < 0.0f || wo.z < 0.0f) return {0, 0,0 }; 
    RGB_F32 albedo = par.albedo;
    return albedo * inv_pi; 
  }
  MPHD MPINL f32     BsdfImpl<BsdfType::Lambertain>::pdf(vec3f wi, vec3f wo, const BsdfParam& par) {
    if(wi.z < 0.0f || wo.z < 0.0f) return 0.0f;
     return wi.z * inv_pi; 
  }
  MPHD MPINL vec3f   BsdfImpl<BsdfType::Lambertain>::sample(vec3f wo, f32& pdf, const BsdfParam& par, Random& rd) {
    if(wo.z < 0.0f) {
      pdf = 0.0f; 
      return vec3f{};  
    }
    f32 r = mps::sqrtf(rd.next1D()), theta = rd.next1D() * mps::pi * 2; 
    auto dir = r * vec2f{mps::cosf(theta), mps::sinf(theta)}; 
    f32 z = mps::sqrtf(mps::maxf(0.0f, 1.0f - r*r)); 
    vec3f wi = vec3f{dir.x, dir.y, z}; 
    pdf = z * mps::inv_pi; 
    return wi; 
  }

}