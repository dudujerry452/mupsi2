#include "core/tracer/trace.h"
#include "core/bsdf/bsdf.h"
#include "core/tracer/intersect.h"
#include "core/math/math_util.h"
#include "core/math/random.h"

namespace mps::core {

  MPHD RGB_F32 trace(const Ray& ray, const RenderContext& ctx) {
    RGB_F32 ret{}; 
    f32 maxx = ray.max; 
    HitRecord max_hit; 
    for(u32 i = 0; i < ctx.scene.num_spheres; i ++) {
      auto hit = intersect(ray, ctx.scene.spheres[i]); 
      if (!hit.hit || hit.t > maxx) continue; 
      maxx = hit.t; 
      max_hit = hit;    
    }
    // vec3f x = vec3f{1.0f, 1.0f, 1.0f} * (mps::dot(hit.ng, vec3f{-1.0, 0.0, 0.0})); 
    //   ret.r = x.x, ret.g = x.y, ret.b = x.z;
    auto& hit = max_hit; 
    
    auto& bsdf = ctx.scene.bsdfs[hit.bsdfId];
    f32 pdf; 
    BsdfParam bsdf_pars; 
    bsdf_pars.type = bsdf.type; 

    
    auto& tex = ctx.scene.textures[bsdf.texID];
    i32 tex_x = mps::mini(tex.width-1, (i32)(tex.width * hit.uv.x)); 
    i32 tex_y = mps::mini(tex.height-1, (i32)(tex.height * hit.uv.y));  
    bsdf_pars.albedo = tex.pixel(tex_x, tex_y); 
    Random rdm(42); 
    vec3f wi = bsdf_sample(-ray.dir, pdf, bsdf_pars, rdm); 
    
    // not to use bsdf for temporary

    vec3f ret_vec = vec3f{bsdf_pars.albedo.r, bsdf_pars.albedo.g, bsdf_pars.albedo.b} * (mps::dot(hit.ng, vec3f{-1.0, 0.0, 0.0})); 

    return RGB_F32{ret_vec.x, ret_vec.y, ret_vec.z};     
  }


}