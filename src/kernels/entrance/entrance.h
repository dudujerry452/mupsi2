#pragma once 

#include "core/basic/core.h"
#include "core/basic/types.h"
#include "core/tracer/trace.h"

#include "core/basic/texture.h"

#include <vector> 

namespace mps::kernels {

  template<Backend B> struct Entrance {
    static void run(); 
  }; 

  template<> struct Entrance<Backend::CUDA> {
    MPH static std::vector<Texture<RGB_F32>> run(const core::RenderContext& ctx); 
  }; 

  template<> struct Entrance<Backend::CPU> {
    MPH static Texture<RGB_F32> run(const core::RenderContext& ctx); 
  };
}