#include "kernels/entrance/dispatch.h"
#include "core/utils/log.h"

namespace mps::kernels {

  // Placeholder — implemented and tested on a CUDA machine later.
  //
  // Plan (data-lifetime buckets, see docs/plan.md):
  //   1. Static upload: cudaMalloc + cudaMemcpy the SceneDevice arrays
  //      (spheres/meshes/vertices/faces/bvhNodes/bvhTriIndices) once.
  //   2. Frame upload: copy Medium/Camera/TraceSettings + the remapped
  //      SceneDevice (device pointers) per frame.
  //   3. Launch one thread per pixel; each thread calls core::trace (MPHD)
  //      with core::make_seed + cam.generateRay, exactly like the CPU shell.
  //   4. Frame download: cudaMemcpy the framebuffer back to host.
  void Entrance<Backend::CUDA>::render(const core::RenderContext&, const core::Camera&, i32,
                                       FrameBuffer<RGB_F32>&) {
    mplog("CUDA render placeholder (implement on a CUDA machine)");
  }

}
