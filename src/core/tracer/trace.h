#pragma once 

#include "core/basic/types.h"
#include "core/scene/scene.h"
#include "core/tracer/intersect.h"

namespace mps::core {

  struct RenderSettings {
    u32 max_bounce; 
  }; 

  struct RenderContext {
    RenderSettings settings; 
    SceneDevice scene; 
  }; 



  MPHD RGB_F32 trace(const Ray ray, const RenderContext& ctx); 

}