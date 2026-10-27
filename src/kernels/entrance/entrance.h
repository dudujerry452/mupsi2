#pragma once 

#include "core/basic/core.h"
#include "core/basic/types.h"
#include "core/tracer/trace.h"

#include "core/basic/texture.h"

namespace mps::kernels {

  template<Backend B> struct Entrance {
    static void run(); 
  }; 

  template<> struct Entrance<Backend::CUDA> {
    static Texture<RGB_F32> run(const core::RenderContext& ctx); 
  }; 

  template<> struct Entrance<Backend::CPU> {
    static Texture<RGB_F32> run(const core::RenderContext& ctx); 
  };
}