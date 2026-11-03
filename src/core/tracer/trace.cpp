#include "core/tracer/trace.h"
#include "core/bsdf/bsdf.h"
#include "core/tracer/intersect.h"
#include "core/math/math_util.h"
#include "core/math/random.h"

namespace mps::core {

  MPHD RGB_F32 trace(const Ray& ray, const RenderContext& ctx, Random& rdm) {

    RGB_F32 ret{}; 
    f32 maxx = ray.max; 
    HitRecord max_hit{}; 
    for(u32 i = 0; i < ctx.scene.num_spheres; i ++) {
      auto hit = intersect(ray, ctx.scene.spheres[i]); 
      if (!hit.hit || hit.t > maxx) continue; 
      maxx = hit.t; 
      max_hit = hit;    
    }

    auto& hit = max_hit; 
    if(hit.hit) {
    
      // # bsdf -> wi, radiance 
      auto& bsdf = ctx.scene.bsdfs[hit.bsdfID];
      f32 pdf; 
      BsdfParam bsdf_pars; 
      bsdf_pars.type = bsdf.type; 
      
      auto& tex = ctx.scene.textures[bsdf.texID];
      i32 tex_x = mps::mini(tex.width-1, (i32)(tex.width * hit.uv.x)); 
      i32 tex_y = mps::mini(tex.height-1, (i32)(tex.height * hit.uv.y));  
      bsdf_pars.albedo = tex.pixel(tex_x, tex_y); 
      
      vec3f wi = bsdf_sample(-ray.dir, pdf, bsdf_pars, rdm); 
      RGB_F32 albedo = bsdf_f(wi, -ray.dir, bsdf_pars); 

      // get wi and radiance
      vec3f ret_vec1 = vec3f{albedo.r, albedo.g, albedo.b} * (mps::dot(hit.ng, vec3f{-1.0, 0.0, 0.0})); 

      // emission 
      if(hit.hasEmis) {
        auto& emis_tex = ctx.scene.textures[hit.emisTexID]; 
        RGB_F32 emis = emis_tex.pixel(tex_x, tex_y); 
        ret_vec1 = ret_vec1 + vec3f{emis.r, emis.g, emis.b}; 
      }

      // # summary 

      vec3f ret_vec = ret_vec1; 
      return RGB_F32{ret_vec.x, ret_vec.y, ret_vec.z};  
    }
    else {
      return RGB_F32{}; 
    }
  }


}