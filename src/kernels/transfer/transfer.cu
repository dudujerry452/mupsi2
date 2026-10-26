#include "core/basic/types.h"
#include "core/primitive/primitive.h"
#include "core/scene/scene.h"
#include "core/utils/cuda_util.h"
#include "kernels/transfer/transfer.h" 
#include <cuda_runtime.h>
#include <driver_types.h>

namespace mps::kernels {

  MPH core::SceneDevice uploadScene(const core::SceneDevice& scene) {
    core::SceneDevice ret = scene; 
    auto dev = upload(scene.spheres, scene.num_spheres);
    ret.spheres = dev;  
    return ret; 
  }

  MPH void destoryScene(core::SceneDevice& scene) {
    destory(scene.spheres); 
    scene.spheres = nullptr; 
  }


  
}