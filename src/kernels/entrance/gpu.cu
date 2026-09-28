#include "kernels/entrance/dispatch.h"
#include "core/utils/log.h"

namespace mps::kernels {

  void Entrance<Backend::CUDA>::run() {
    mplog("Use Backend: CUDA");

  }

}