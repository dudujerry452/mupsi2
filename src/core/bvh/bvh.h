#pragma once 

#include "core/basic/types.h"
#include "core/scene/scene.h"


namespace mps::core {

  struct BVHNode {
    AABB aabb; 

    u32 left; 
    u32 right; 

    u32 first; 
    u32 count; 
  }; 

  MPH void buildBLAS(u32 meshID, SceneDevice& scene);
  MPH void buildTLAS(SceneDevice& scene);  

}