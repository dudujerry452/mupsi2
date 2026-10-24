#include "core/primitive/primitive.h"

namespace mps::core {

  MPHD Sphere buildSphere(vec3f pos, f32 r) {
    Sphere ret{}; 
    ret.info.type = PrimitiveType::Sphere; 
    ret.info.wpos = pos; 
    ret.info.scale = 1.0f; 
    ret.radius = r; 
    return ret; 
  }

}