#pragma once 

#include "core/basic/types.h"

namespace mps::kernels {

  template<Backend B> struct test_func; 
  
  template<> struct test_func<Backend::CUDA> {
    static void run(i32 n, u32* out); 
  }; 
  template<> struct test_func<Backend::CPU> {
    static void run(i32 n, u32* out); 
  }; 
}