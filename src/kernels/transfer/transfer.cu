#include "core/primitive/primitive.h"
#include "core/utils/cuda_util.h"
#include "kernels/transfer/transfer.h" 
#include <driver_types.h>

namespace mps::kernels {

  MPH void upload(core::SceneDevice& scene) {

    u32 n = scene.num_spheres;

    if (n != 0 && scene.spheres != nullptr) {
      u32 bytes = n * sizeof(core::Sphere); 

      core::Sphere *dev_sphere = nullptr; 
      cudaCheck(cudaMalloc(&dev_sphere, bytes));
      cudaCheck(cudaMemcpy(dev_sphere, scene.spheres, bytes, cudaMemcpyHostToDevice)); 
      
      //rebind 

      delete[] scene.spheres; 
      scene.spheres = dev_sphere; 
    }
    n = 0; 
  
  }

  MPH void destory(const core::SceneDevice& dev) {
    cudaCheck(cudaFree(dev.spheres)); 
  }

  
}