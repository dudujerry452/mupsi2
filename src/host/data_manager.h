#pragma once 

#include "core/basic/types.h"
#include <tuple>
#include <vector>

namespace mps::host {

  // return starts(heap) flat(bytes)
  std::tuple<u32, u32*, void*> flatten_primitive(const std::vector<void*>& prims); 
  

}