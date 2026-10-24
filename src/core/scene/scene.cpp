#include "core/scene/scene.h"
#include "core/primitive/primitive.h"

namespace mps::core {

  MPH core::SceneDevice transform(const core::SceneHost& host) {
    core::SceneDevice dev{};

    u32 cnt = host.spheres.size(); 
    if(cnt != 0) {
      dev.spheres = new core::Sphere[cnt]();
      std::copy(host.spheres.begin(), host.spheres.end(), dev.spheres);
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