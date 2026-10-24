#include "kernels/entrance/dispatch.h"
#include "kernels/transfer/transfer.h"

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


    // core::trace(const Ray ray, ctx); 



  }

  void Entrance<Backend::CUDA>::run(const core::RenderContext& ctx) {
    mplog("Use Backend: CUDA");

    core::RenderContext _ctx = ctx; 
    upload(_ctx.scene);

    dim3 blockdim(16, 16); 
    dim3 griddim((ctx.framebuffer->width + blockdim.x-1) / blockdim.x, 
  (ctx.framebuffer->height + blockdim.y-1) / blockdim.y); 

    trace<<<griddim, blockdim>>>(_ctx); 

    // get data back

    destory(_ctx.scene);  
  }

}