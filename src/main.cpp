#include <iostream>
#include <vector>

#include "core/core.h"
#include "kernels/entrance/dispatch.h"

#if defined(MUPSI_USE_CUDA)
constexpr mps::Backend ACTIVE_BACKEND = mps::Backend::CUDA;
#else
constexpr mps::Backend ACTIVE_BACKEND = mps::Backend::CPU;
#endif

int main() {

    mps::kernels::Entrance<ACTIVE_BACKEND>::run(); 
    return 0;
}
