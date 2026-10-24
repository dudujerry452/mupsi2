#pragma once 

#include "core/basic/core.h"
#include <string>

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

  MPHD Sphere buildSphere(vec3f pos, f32 r);

  struct Mesh {
    PrimitiveInfo info; 
    vec3f* vertices; 
    u32* indexes; 
  };

  MPH Mesh buildMesh(std::string path); 

}