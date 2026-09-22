#include "kernels/entrance/dispatch.h"
#include "utils/log.inl"

namespace mps::kernels {

  void Entrance<Backend::CPU>::run() {
    llog("Use Backend: CPU");

  }

}