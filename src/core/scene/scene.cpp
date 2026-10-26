#include "core/scene/scene.h"
#include "core/primitive/primitive.h"

#include "core/utils/cuda_util.h"

namespace mps::core {

  
  MPH void SceneDevice::upload() {
    auto dev = ::mps::upload(spheres, num_spheres);
    spheres = dev;  
  }
  MPH void SceneDevice::destory() {
    ::mps::destory(spheres); 
    *this = SceneDevice{};
  }


  MPH void SceneDevice::free() {
    if(spheres != nullptr) delete[] spheres; 
    *this = SceneDevice{}; 
  }

  MPH SceneDevice SceneHost::transform() const {
    core::SceneDevice dev{};

    dev.num_spheres = spheres.size(); 
    if(dev.num_spheres != 0) {
      dev.spheres = new core::Sphere[dev.num_spheres]();
      std::copy(spheres.begin(), spheres.end(), dev.spheres);
    }
    return dev; 
  }

  MPH void SceneHost::AddSphere(Sphere&& sphere) {
    spheres.push_back(std::move(sphere)); 
  }

  MPH SceneHost buildScene_Spot() {
    SceneHost scene; 
    scene.AddSphere(buildSphere({0, 0, -10}, 2));
    return scene; 
  }

}