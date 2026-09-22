#pragma once 

#include "core/basic/core.h"

namespace mps {

  enum class BsdfType: u32 {
    Lambertain, 
    Specular
  }; 

  template<BsdfType T>
  struct Bsdf; 
  
  template<>
  struct Bsdf<BsdfType::Lambertain>{
    BsdfType type; 
  }; 

}