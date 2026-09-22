#pragma once 

#include "core/basic/core.h"

namespace mps::kernels {

  template<Backend B> struct Entrance {
    static void run(); 
  }; 

  template<> struct Entrance<Backend::CUDA> {
    static void run(); 
  }; 


  template<> struct Entrance<Backend::CPU> {
    static void run(); 
  };
}