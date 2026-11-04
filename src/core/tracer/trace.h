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

  struct debug_render_lines {
    struct line {
      vec3f start; 
      vec3f end; 
      RGB_F32 albedo; 
    }; 
    line lines[50]; // max 50 bounces
    u32 num_bounce; 

    debug_render_lines* dev; // 设备副本句柄(upload/download/destory 管理)

  }; 

  MPHD debug_render_lines trace_visual(const Ray& ray, const RenderContext& ctx, Random& rdm);

}