#pragma once 

#include "core/basic/core.h"
#include "core/basic/types.h"
#include "core/tracer/trace.h"

#include "core/tracer/framebuffer.h"

namespace mps::kernels {

  template<Backend B> struct Entrance {
    static void run(); 
  }; 

  template<> struct Entrance<Backend::CUDA> {
    static FrameBuffer<RGB_F32> run(const core::RenderContext& ctx); 
  }; 

  template<> struct Entrance<Backend::CPU> {
    static FrameBuffer<RGB_F32> run(const core::RenderContext& ctx); 
  };
}