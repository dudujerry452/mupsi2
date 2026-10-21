#pragma once

#include "core/basic/core.h"
#include "core/tracer/trace.h"

namespace mps::kernels {

  // Backend shell: renders `ctx` with `spp` samples per pixel into `fb`.
  template<Backend B> struct Entrance {
    static void render(const core::RenderContext& ctx, const core::Camera& cam, i32 spp,
                       FrameBuffer<RGB_F32>& fb);
  };

  template<> struct Entrance<Backend::CPU> {
    static void render(const core::RenderContext& ctx, const core::Camera& cam, i32 spp,
                       FrameBuffer<RGB_F32>& fb);
  };

  template<> struct Entrance<Backend::CUDA> {
    static void render(const core::RenderContext& ctx, const core::Camera& cam, i32 spp,
                       FrameBuffer<RGB_F32>& fb);
  };

}
