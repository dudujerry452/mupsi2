#pragma once 

#include <vector> 
#include "core/basic/types.h"
#include "core/primitive/primitive.h"
#include "core/basic/texture.h"
#include "core/bsdf/bsdf.h"

namespace mps::core {



 
  

   struct SceneDevice {
    
    u32 num_vertices;
    u32 num_indexes;  
    vec3f *vertices; 
    u32 *indexes; 

    u32 num_spheres; 
    Sphere *spheres; 

    u32 num_textures; 
    Texture<RGB_F32> *textures; 

    u32 num_bsdfs; 
    Bsdf *bsdfs; 

    // ---- cuda ----
    // upload to device and rebind
    MPH void upload(); 
    MPH void destory();  

    // ---- cpu ----
    MPH void free(); 
  }; 

  struct SceneHost {

    std::vector<Mesh> meshes; 
    std::vector<Sphere> spheres;  
    // std::vector<void*> lights; 
    std::vector<Texture<RGB_F32> > textures;
    std::vector<Bsdf> bsdfs; 
    // std::vector<Medium> media;  

    MPH u32 AddSphere(Sphere&& sphere); 
    MPH u32 AddTexture(const char* path); 
    MPH u32 AddTexture(i32 width, i32 height); 
    MPH u32 AddBsdf(BsdfType type, u32 texID); 
    MPH core::SceneDevice transform() const; 
  }; 

  MPH SceneHost buildScene_Spot(); 

  
}