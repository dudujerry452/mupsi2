#include "core/tracer/trace.h"
#include "core/basic/types.h"
#include "core/bsdf/bsdf.h"
#include "core/scene/scene.h"
#include "core/tracer/intersect.h"
#include "core/math/math_util.h"
#include "core/math/random.h"

namespace mps::core {

  // scene intersect 
  MPHD HitRecord intersect(const Ray& ray, const SceneDevice& scene) {
    f32 maxx = ray.max; 
    HitRecord max_hit{}; 
    for(u32 i = 0; i < scene.num_spheres; i ++) {
      auto hit = intersect(ray, scene.spheres[i]); 
      if (!hit.hit || hit.t > maxx) continue; 
      maxx = hit.t; 
      max_hit = hit;    
    }
    return max_hit; 
  }

  struct bsdf_ret {
    vec3f wi; 
    vec3f wo; 
    f32 pdf; 
    RGB_F32 albedo; 
    RGB_F32 emission; 
  }; 

  // assume hit.hit = true
  MPHD bsdf_ret compute_bsdf(const HitRecord &hit, vec3f wo, const SceneDevice& scene, Random& rdm) {

    bsdf_ret ret{}; 

    auto& bsdf = scene.bsdfs[hit.bsdfID];
    f32 pdf; 
    BsdfParam bsdf_pars; 
    bsdf_pars.type = bsdf.type; 
    
    auto& tex = scene.textures[bsdf.texID];
    i32 tex_x = mps::mini(tex.width-1, (i32)(tex.width * hit.uv.x)); 
    i32 tex_y = mps::mini(tex.height-1, (i32)(tex.height * hit.uv.y));  
    bsdf_pars.albedo = tex.pixel(tex_x, tex_y); 
    
    vec3f wi = bsdf_sample(wo, pdf, bsdf_pars, rdm); 
    ret.albedo = RGB_F32(bsdf_f(wi, wo, bsdf_pars)); 

    // emission 
    if(hit.hasEmis) {
      auto& emis_tex = scene.textures[hit.emisTexID]; 
      ret.emission = emis_tex.pixel(tex_x, tex_y); 
    }

    ret.pdf = pdf;
    ret.wi = wi; 
    ret.wo = wo;  
    return ret; 
  }

  MPHD RGB_F32 trace(const Ray& ray, const RenderContext& ctx, Random& rdm) {

    RGB_F32 ret{}; 
    
    auto hit = intersect(ray, ctx.scene); 
    if(hit.hit) {
      auto bsdf_ret = compute_bsdf(hit, -ray.dir, ctx.scene, rdm); 
      return bsdf_ret.albedo * (mps::dot(hit.ng, vec3f{-1.0, 0.0, 0.0})) + bsdf_ret.emission;  
    }
    else {
      return RGB_F32{}; 
    }
  }


}