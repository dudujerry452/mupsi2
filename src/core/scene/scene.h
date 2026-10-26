#pragma once 

#include <vector> 
#include "core/basic/types.h"
#include "core/primitive/primitive.h"

namespace mps::core {



  struct SceneDevice {
    
    u32 num_vertices;
    u32 num_indexes;  
    vec3f* vertices; 
    u32* indexes; 

    u32 num_spheres; 
    Sphere* spheres; 
  }; 

  struct SceneHost {

    std::vector<Mesh> meshes; 
    std::vector<Sphere> spheres;  
    // std::vector<void*> lights; 
    // std::vector<Texture> textures;
    // std::vector<Bsdf> bsdfs; 
    // std::vector<Medium> media;  

    MPH void AddSphere(Sphere&& sphere); 
  }; 

  MPH core::SceneDevice transform(const core::SceneHost& host); 
  MPH SceneHost buildScene_Spot(); 
}