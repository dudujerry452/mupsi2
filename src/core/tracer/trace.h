#pragma once 

#include "core/basic/types.h"
#include "core/scene/scene.h"
#include "core/scene/camera.h"
#include "core/tracer/intersect.h"

#include "core/basic/texture.h"

#include "core/tracer/constant.h"

namespace mps::core {

  struct RenderSettings {
    i32 max_bounce; 
  }; 

  struct RenderContext {
    RenderSettings settings; 
    RenderConstants constant; 
    SceneDevice scene; 
    Camera camera; 
    Texture<RGB_F32> framebuffer; 
    Texture<RGB_F32> framebuffer_1;
    Texture<RGB_F32> framebuffer_2;  
  }; 


  MPHD RGB_F32 trace(const Ray& ray, const RenderContext& ctx, Random& rdm); 
  MPHD RGB_F32 trace_depth(const Ray& ray, const RenderContext& ctx, Random& rdm);

  struct debug_render_lines; 
  MPHD debug_render_lines trace_visual(const Ray& ray, const RenderContext& ctx, Random& rdm);

}