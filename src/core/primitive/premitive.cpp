#include "core/primitive/primitive.h"

namespace mps::core {

  MPHD Sphere buildSphere(vec3f pos, f32 r, u32 bsdfID) {
    Sphere ret{}; 
    ret.info.type = PrimitiveType::Sphere; 
    ret.info.wpos = pos;
    ret.info.quater = vec4f{0, 0, 0, 1};  
    ret.info.scale = 1.0f; 
    ret.info.bsdfID = bsdfID; 
    ret.radius = r; 
    return ret; 
  }

}