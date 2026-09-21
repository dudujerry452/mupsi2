#include "kernels/entrance/dispatch.h"
#include "core/log.inl"

namespace mps::kernels {

  void Entrance<Backend::CUDA>::run() {
    llog("Use Backend: CUDA");

  }

}