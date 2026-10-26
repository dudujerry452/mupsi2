#include "core/tracer/trace.h"
#include "kernels/entrance/entrance.h"
#include "core/utils/log.h"

namespace mps::kernels {

  FrameBuffer<RGB_F32> Entrance<Backend::CPU>::run(const core::RenderContext& ctx) {
    mplog("Use Backend: CPU");

    return FrameBuffer<RGB_F32>{};

  }

}