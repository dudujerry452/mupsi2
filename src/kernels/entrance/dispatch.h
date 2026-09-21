#pragma once 

#include "core/core.inl"

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