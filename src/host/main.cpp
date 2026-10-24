#include <iostream>
#include <vector>

#include "core/basic/core.h"
#include "core/scene/scene.h"
#include "core/tracer/trace.h"
#include "kernels/entrance/dispatch.h"


#include "host/ppm_util.h"

#if defined(MUPSI_USE_CUDA)
constexpr mps::Backend ACTIVE_BACKEND = mps::Backend::CUDA;
#else
constexpr mps::Backend ACTIVE_BACKEND = mps::Backend::CPU;
#endif

int main() {

    mps::core::RenderContext ctx; 
    ctx.scene = mps::core::transform(mps::core::buildScene_Spot()); 

    mps::kernels::Entrance<ACTIVE_BACKEND>::run(ctx); 


    
    return 0;
}
