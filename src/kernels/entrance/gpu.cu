#include "core/math/math_util.h"
#include "kernels/entrance/entrance.h"

#include "core/primitive/primitive.h"
#include "core/tracer/trace.h"

#include "core/utils/log.h"
#include "core/scene/scene.h"
#include "core/utils/cuda_util.h"

namespace mps::kernels {


  // MPH core::SceneDevice* transfer(const core::SceneDevice* host_dev) {
  //   core::SceneDevice* dev = nullptr; 

  //   cudaCheck(cudaMalloc(&dev, sizeof(core::SceneDevice)));
  //   cudaCheck(cudaMemcpy(dev, host_dev, 1, ::cudaMemcpyHostToDevice)); 
  // }

  MPG void trace(core::RenderContext ctx) {
    u32 col = blockIdx.x * blockDim.x + threadIdx.x; 
    u32 row = blockIdx.y * blockDim.y + threadIdx.y; 

    u32 width = ctx.framebuffer.width; 
    u32 height = ctx.framebuffer.height; 

    f32 x = ((f32)col + 0.5f) / width, y = ((f32)row+0.5f) / height;
    
    f32 real_height = ctx.camera.znear * mps::tanf(ctx.camera.fov * 0.5) * 2; 
    f32 real_width = ctx.camera.aspect * real_height; 

    core::Ray ray{}; 
    ray.o = ctx.camera.wpos; 
    ray.dir = vec3f{real_width * x - 0.5f * real_width, real_height * y - 0.5f * real_height, -1};
    ray.dir = quat_rotate(ctx.camera.quat, ray.dir); 
    ray.dir = mps::normalize(ray.dir); 

    ray.min = 0.01f; 
    ray.max = 10000.0f; 

    ctx.framebuffer.pixel(col, row) = core::trace(ray, ctx); 
  }

  FrameBuffer<RGB_F32> Entrance<Backend::CUDA>::run(const core::RenderContext& ctx) {
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

    auto ret = _ctx.framebuffer; 
    ret.download();  
    _ctx.framebuffer.destory(); 
    _ctx.scene.destory(); 
    return ret;   
  }

}