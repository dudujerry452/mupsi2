#include "core/tracer/trace.h"
#include "kernels/entrance/dispatch.h"
#include "core/utils/log.h"

namespace mps::kernels {

  void Entrance<Backend::CPU>::run(const core::RenderContext& ctx) {
    mplog("Use Backend: CPU");

  }

}