#include "core/basic/texture.h"
#include "core/math/math_util.h"
#include "core/tracer/constant.h"
#include "kernels/entrance/entrance.h"

#include "core/primitive/primitive.h"
#include "core/tracer/trace.h"

#include "core/utils/log.h"
#include "core/scene/scene.h"
#include "core/utils/cuda_util.h"

#include "core/math/random.h"

#include <vector>

namespace mps::kernels {


  MPHD core::Ray cast_ray(u32 col, u32 row, const core::RenderContext& ctx) {
    u32 width = ctx.framebuffer.width; 
    u32 height = ctx.framebuffer.height; 

    f32 x = ((f32)col + 0.5f) / width, y = 1.0f - ((f32)row+0.5f) / height;
    
    f32 real_height = ctx.camera.znear * mps::tanf(ctx.camera.fov * 0.5) * 2; 
    f32 real_width = ctx.camera.aspect * real_height; 

    core::Ray ray{}; 
    ray.o = ctx.camera.wpos; 
    ray.dir = vec3f{real_width * x - 0.5f * real_width, real_height * y - 0.5f * real_height, -ctx.camera.znear};
    ray.dir = quat_rotate(ctx.camera.quat, ray.dir); 
    ray.dir = mps::normalize(ray.dir); 

    ray.min = core::RenderConstants::RAY_MIN; 
    ray.max = ctx.constant.RAY_MAX; 

    

    return ray; 
  }

  MPG void trace(core::RenderContext ctx) {
    u32 col = blockIdx.x * blockDim.x + threadIdx.x; 
    u32 row = blockIdx.y * blockDim.y + threadIdx.y; 

    core::Ray ray = cast_ray(col, row, ctx); 
    mps::core::Random rdm; rdm.init(mps::core::make_seed_f(col, row)); 

    ctx.framebuffer.pixel(col, row) = core::trace(ray, ctx, rdm); 
  }

  MPG void trace_depth(core::RenderContext ctx) {
    u32 col = blockIdx.x * blockDim.x + threadIdx.x; 
    u32 row = blockIdx.y * blockDim.y + threadIdx.y; 

    core::Ray ray = cast_ray(col, row, ctx); 
    mps::core::Random rdm; rdm.init(mps::core::make_seed_f(col, row)); 

    ctx.framebuffer.pixel(col, row) = core::trace_depth(ray, ctx, rdm); 
  }

  MPG void trace_lines(core::Ray ray, core::RenderContext ctx) {
    u32 col = blockIdx.x * blockDim.x + threadIdx.x; 
    u32 row = blockIdx.y * blockDim.y + threadIdx.y; 

    if(col == 0 && row == 0) {
      // TODO: use trace_visual drawing a ray stack, and download to host somehow
    }
  }

  std::vector<Texture<RGB_F32>> Entrance<Backend::CUDA>::run(const core::RenderContext& ctx) {
    mplog("Use Backend: CUDA");

    auto _ctx = ctx; 

    _ctx.scene.upload(); 
    _ctx.framebuffer.upload(); 

    dim3 blockdim(16, 16); 
    dim3 griddim(
      (_ctx.framebuffer.width  + blockdim.x-1) / blockdim.x, 
      (_ctx.framebuffer.height + blockdim.y-1) / blockdim.y
    ); 

    trace<<<griddim, blockdim>>>(_ctx); 
    cudaCheckLast();

    std::vector<Texture<RGB_F32>> ret; 
    auto res = _ctx.framebuffer; 
    res.download();
    ret.push_back(res);   

    trace_depth<<<griddim, blockdim>>>(_ctx); 
    cudaCheckLast();
    res = _ctx.framebuffer; 
    res.download();
    ret.push_back(res); 

    _ctx.framebuffer.destory(); 
    _ctx.scene.destory(); 
    return ret;   
  }

}