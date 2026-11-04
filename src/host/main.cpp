#include <iostream>
#include <new>
#include <vector>

#include "core/basic/core.h"
#include "core/basic/types.h"
#include "core/scene/scene.h"
#include "core/tracer/trace.h"
#include "kernels/entrance/entrance.h"


#include "host/ppm_util.h"

#if defined(MUPSI_USE_CUDA)
constexpr mps::Backend ACTIVE_BACKEND = mps::Backend::CUDA;
#else
constexpr mps::Backend ACTIVE_BACKEND = mps::Backend::CPU;
#endif

int main() {

    mps::core::RenderContext ctx{}; 
    ctx.scene = mps::core::buildScene_Spot().transform(); 
    ctx.camera.fov = 60.0f * mps::pi / 180.0f;  
    ctx.camera.znear = 1.0f; 
    ctx.camera.aspect = 2.0f; 
    ctx.camera.LookAt({0, 0, -1}, {0, 1, 0}); 
    ctx.framebuffer.width = 512; 
    ctx.framebuffer.height = 256; 
    ctx.framebuffer.alloc(); 
    ctx.framebuffer_1.width = 512; 
    ctx.framebuffer_1.height = 256; 
    ctx.framebuffer_1.alloc(); 
    ctx.framebuffer_2.width = 512; 
    ctx.framebuffer_2.height = 256; 
    ctx.framebuffer_2.alloc(); 
    ctx.settings.max_bounce = 10; 

    auto fbs = mps::kernels::Entrance<ACTIVE_BACKEND>::run(ctx); 
    
    mps::Texture<mps::RGB_8Bit> out; 
    out.width = ctx.framebuffer.width; 
    out.height = ctx.framebuffer.height;  
    out.alloc(); 

    mps::host::radianceTo8bit(out, fbs[0]); 
    mps::host::savePPM("test.ppm", out); 
    mps::host::radianceTo8bit(out, fbs[1]); 
    mps::host::savePPM("test_depth.ppm", out); 
    mps::host::radianceTo8bit(out, fbs[2]); 
    mps::host::savePPM("test_debug.ppm", out); 

    out.free();
    fbs[0].free();
    fbs[1].free();  
    fbs[2].free();  
    ctx.framebuffer.free(); 
    ctx.framebuffer_1.free(); 
    ctx.framebuffer_2.free(); 
    ctx.scene.free(); 
    
    return 0;
}
