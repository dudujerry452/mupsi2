#include "core/tracer/trace.h"
#include "core/tracer/intersect.h"
#include "core/math/math_util.h"

namespace mps::core {

  MPHD RGB_F32 trace(const Ray ray, const RenderContext& ctx) {
    RGB_F32 ret{}; 
    for(u32 i = 0; i < ctx.scene.num_spheres; i ++) {
      auto hit = intersect(ray, ctx.scene.spheres[i]); 
      vec3f x = vec3f{1.0f, 1.0f, 1.0f} * (mps::dot(hit.ng, vec3f{-1.0, 0.0, 0.0})); 
      ret.r = x.x, ret.g = x.y, ret.b = x.z; 
    }
    return ret;     
  }


}