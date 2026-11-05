#pragma once 

#include "core/basic/core.h"
#include <string>
#include <vector>

namespace mps::core {

  enum class PrimitiveType {
    Sphere, 
    Mesh 
  }; 

  struct alignas(16) PrimitiveInfo {
    PrimitiveType type; 
    u16 bsdfID;
    byte hasEmis; 
    u16 emisTexID;
    vec3f wpos;
    vec4f quater; 
    f32 scale; 
  }; 

  struct Primitive {
    PrimitiveType type; 
    u32 primID; 
  }; 

  struct Sphere {
    PrimitiveInfo info; 
    f32 radius; 
  }; 

  MPHD Sphere buildSphere(vec3f pos, f32 r, u32 bsdfID);

  struct MeshDevice; 
  struct SceneDevice; 

  struct MeshHost {
    PrimitiveInfo info; 
    
    std::vector<vec3f> vertices; 
    std::vector<vec3f> normals; 
    std::vector<u32> indices; 
    std::vector<vec2f> uvs; 

    // alloc and add a MeshDevice to SceneDevice on host
    MPH void transform(SceneDevice& scene) const; 
  };

  struct MeshDevice {
    PrimitiveInfo info; 

    u32 num_vertices; 
    // vec3f* vertices; 
    // vec3f* normals; 
    // vec2f* uvs; 
    u32 offset_vertices; 

    u32 num_indices; 
    // u32* indices; 
    u32 offset_indices; 
  }; 

  MPH MeshHost buildMeshHost(std::string path); 

}