#pragma once 

#include <vector> 
#include "core/basic/types.h"
#include "core/primitive/primitive.h"

namespace mps::core {

  struct SceneHost {

    std::vector<void*> primitives; 
    // std::vector<void*> lights; 
    // std::vector<Texture> textures;
    // std::vector<Bsdf> bsdfs; 
    // std::vector<Medium> media;  

  }; 

  struct SceneDevice {
    u32 num; 
    u32* starts; 
    u32* lens; 
    void* primitives; 

    template<PrimitiveType T> 
    MPHD MPINL Primitive<T>* get_primitive(u32 id) {
      if (id >= num) return nullptr; 
      u32 pos = starts[id]; 
      return reinterpret_cast<Primitive<T>>(static_cast<char*>(primitives) + pos); 
    }
 

    

  }; 

}