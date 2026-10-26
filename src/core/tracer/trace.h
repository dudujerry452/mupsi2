#pragma once 

#include "core/basic/types.h"
#include "core/scene/scene.h"
#include "core/scene/camera.h"
#include "core/tracer/intersect.h"

#include "core/tracer/framebuffer.h"

namespace mps::core {

  struct RenderSettings {
    u32 max_bounce; 
  }; 

  struct RenderContext {
    RenderSettings settings; 
    SceneDevice scene; 
    Camera camera; 
    FrameBuffer<RGB_F32> framebuffer; 
  }; 


  MPHD RGB_F32 trace(const Ray& ray, const RenderContext& ctx); 

}