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

#include "host/draw_debug.h"

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

  MPG void trace_lines(core::RenderContext ctx, core::debug_render_lines* out) {
    u32 col = blockIdx.x * blockDim.x + threadIdx.x; 
    u32 row = blockIdx.y * blockDim.y + threadIdx.y; 

    u32 width = ctx.framebuffer.width; 
    u32 height = ctx.framebuffer.height; 
    if (col >= width || row >= height) return; 
    if (col % 10 != 0 || row % 10 != 0) return; // 每 10x10 像素画一次

    u32 stride = (width + 9) / 10; 
    u32 idx = (row / 10) * stride + (col / 10); 

    core::Ray ray = cast_ray(col, row, ctx); 
    mps::core::Random rdm; rdm.init(mps::core::make_seed_f(col, row)); 
    out[idx] = core::trace_visual(ray, ctx, rdm); // 世界坐标, 投影在 host draw_debug 做
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

    // ---- debug ray path: 每 10x10 像素画一条路径 ----
    u32 width = _ctx.framebuffer.width; 
    u32 height = _ctx.framebuffer.height; 
    u32 stride = (width + 9) / 10; 
    u32 nrows = (height + 9) / 10; 
    u32 count = stride * nrows; 

    core::debug_render_lines* host_paths = new core::debug_render_lines[count](); 
    core::debug_render_lines* dev_paths = ::mps::upload(host_paths, count); 
    delete[] host_paths; 

    trace_lines<<<griddim, blockdim>>>(_ctx, dev_paths); 
    cudaCheckLast();

    core::debug_render_lines* paths = ::mps::download(dev_paths, count); 
    ::mps::destory(dev_paths); 

    // overlay the ray path onto a copy of the rendered image
    Texture<RGB_F32> overlay; 
    overlay.width = ret[0].width; 
    overlay.height = ret[0].height; 
    overlay.alloc(); 
    for (i32 i = 0; i < ret[0].pixelnum(); ++i) overlay.pixels[i] = ret[0].pixels[i]; 
    host::draw_debug(overlay, ret[1], paths, count, ctx.camera); 
    delete[] paths; 
    ret.push_back(overlay); 

    _ctx.framebuffer.destory(); 
    _ctx.scene.destory(); 
    return ret;   
  }

}