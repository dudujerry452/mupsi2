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
    ctx.scene = mps::core::transform(mps::core::buildScene_Spot()); 
    ctx.camera.fov = 60.0f * mps::pi / 180.0f;  
    ctx.camera.znear = 1.0f; 
    ctx.camera.aspect = 2.0f; 
    ctx.camera.LookAt({0, 0, -1}, {0, 1, 0}); 
    ctx.framebuffer.width = 512; 
    ctx.framebuffer.height = 256; 
    ctx.framebuffer.pixels = new mps::RGB_F32[ctx.framebuffer.pixelnum()]();

    auto fb = mps::kernels::Entrance<ACTIVE_BACKEND>::run(ctx); 
    
    mps::FrameBuffer<mps::RGB_8Bit> out; 
    out.width = ctx.framebuffer.width; 
    out.height = ctx.framebuffer.height;  
    out.pixels = new mps::RGB_8Bit[out.pixelnum()]();  

    mps::host::radianceTo8bit(out, fb); 
    mps::host::savePPM("test.ppm", out); 
    
    delete[] fb.pixels; 
    delete[] out.pixels; 
    delete[] ctx.framebuffer.pixels; 
    delete[] ctx.scene.spheres; 
    
    return 0;
}
