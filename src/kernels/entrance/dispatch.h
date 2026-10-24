#pragma once 

#include "core/basic/core.h"
#include "core/tracer/trace.h"

namespace mps::kernels {

  template<Backend B> struct Entrance {
    static void run(); 
  }; 

  template<> struct Entrance<Backend::CUDA> {
    static void run(const core::RenderContext& ctx); 
  }; 

  template<> struct Entrance<Backend::CPU> {
    static void run(const core::RenderContext& ctx); 
  };
}