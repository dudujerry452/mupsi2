#include "kernels/entrance/dispatch.h"
#include "core/utils/log.h"

namespace mps::kernels {

  void Entrance<Backend::CPU>::run() {
    mplog("Use Backend: CPU");

  }

}