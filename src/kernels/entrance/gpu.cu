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


  void Entrance<Backend::CUDA>::run(const core::RenderContext& ctx) {
    mplog("Use Backend: CUDA");

    core::RenderContext _ctx = ctx; 
    upload(_ctx.scene);

  }

}