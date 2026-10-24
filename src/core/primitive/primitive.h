#pragma once 

#include "core/basic/core.h"
#include "core/utils/panic.h"


namespace mps::core {

  enum class PrimitiveType {
    Sphere, 
    Mesh 
  }; 

  struct alignas(16) PrimitiveInfo {
    PrimitiveType type; 
    u16 bsdfId;
    u16 texId;
    vec3f wpos;
    vec4f quater; 
    f32 scale; 
  }; 

  struct Sphere {
    PrimitiveInfo info; 
    f32 radius; 
  }; 

  struct Mesh {
    PrimitiveInfo info; 
  
  }; 

}