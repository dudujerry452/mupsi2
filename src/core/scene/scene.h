#pragma once 

#include <vector> 
#include "core/basic/types.h"

namespace mps::core {

  struct SceneCPU {

    std::vector<void*> primitives; 
    std::vector<void*> lights; 
    // std::vector<Texture> textures;
    // std::vector<Bsdf> bsdfs; 
    // std::vector<Medium> media;  

  }; 

}