#pragma once 

#include "core/basic/types.h"
#include "core/scene/scene.h"
#include <tuple>
#include <vector>

namespace mps::host {

  // return starts(heap) flat(bytes)
  MPH std::tuple<u32, u32*, u32*, void*> flatten_primitive(const std::vector<void*>& prims); 

  // notice: return **host** space SceneDevice
  MPH core::SceneDevice SceneHost2Device(const core::SceneHost& scene); 
  

}