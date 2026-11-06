// this header 

#pragma once 

#include "core/basic/types.h"

namespace mps::core {


  struct RenderConstants {
    static constexpr f32 RAY_MIN = 0.0001f;
    f32 RAY_MAX = 10000.0f;  // should change when the scene change  
    

    static constexpr f32 TBN_T_MAX = 0.9999f; 

    static constexpr u32 MIN_BVH_NODE_COUNT = 4; 
  }; 


}