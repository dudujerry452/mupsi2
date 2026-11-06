#pragma once 

#include "core/basic/types.h"
#include "core/scene/scene.h"


namespace mps::core {



  MPH void buildBLAS(u32 meshID, SceneDevice& scene);
  MPH void buildTLAS(SceneDevice& scene);  

}