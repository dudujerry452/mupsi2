#include "kernels/entrance/dispatch.h"

#include "core/utils/cuda_util.h"
#include "core/utils/log.h"
#include "core/math/random.h"

#include <cuda_runtime.h>

namespace mps::kernels {

  namespace {

    // Upload a host POD array to device memory. Returns nullptr for empty arrays.
    template <typename T>
    T* upload(const T* src, u32 count) {
      if (count == 0 || src == nullptr) return nullptr;
      T* d = nullptr;
      cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d), size_t(count) * sizeof(T)));
      cudaCheck(cudaMemcpy(d, src, size_t(count) * sizeof(T), cudaMemcpyHostToDevice));
      return d;
    }

    // One thread per pixel; spp loop is inside the thread (matches the CPU shell).
    __global__ void renderKernel(core::SceneDevice scene, core::Medium medium, bool hasMedium,
                                 core::TraceSettings settings, core::EnvMap envMap,
                                 core::Camera cam, i32 spp, RGB_F32* fb) {
      const i32 x = int(blockIdx.x * blockDim.x + threadIdx.x);
      const i32 y = int(blockIdx.y * blockDim.y + threadIdx.y);
      if (x >= cam.width || y >= cam.height) return;

      core::RenderContext ctx{scene, medium, hasMedium, settings, envMap};

      RGB_F32 acc{0.0f, 0.0f, 0.0f};
      for (i32 s = 0; s < spp; ++s) {
        u32 seed = core::make_seed(x, y, s, 42);
        core::Ray r = cam.generateRay(x, y);
        acc = acc + core::trace(r, ctx, seed);
      }
      fb[y * cam.width + x] = acc;
    }

  } // namespace

  void Entrance<Backend::CUDA>::render(const core::RenderContext& ctx, const core::Camera& cam,
                                       i32 spp, FrameBuffer<RGB_F32>& fb) {
    const core::SceneDevice& h = ctx.scene;
    const i32 w = fb.width, hh = fb.height;

    // ——— 1. static upload: scene geometry (once) ———
    core::Sphere* d_spheres       = upload(h.spheres, h.numSpheres);
    core::Mesh*   d_meshes        = upload(h.meshes, h.numMeshes);
    vec3f*        d_vertices      = upload(h.vertices, h.numVertices);
    u32*          d_faces         = upload(h.faces, h.numFaces);
    core::BVHNode* d_bvhNodes     = upload(h.bvhNodes, h.numBvhNodes);
    u32*          d_bvhTriIndices = upload(h.bvhTriIndices, h.numBvhTriIndices);
    vec3f*        d_vertexNormals = upload(h.vertexNormals, h.numVertices);
    u8*           d_voxelSignGrid  = upload(h.voxelSignGrid, h.voxelRes * h.voxelRes * h.voxelRes);
    u32*          d_voxelBandIndex = upload(h.voxelBandIndex, h.voxelBandCount);
    f32*          d_voxelBandValue = upload(h.voxelBandValue, h.voxelBandCount);
    core::AreaLight* d_lights      = upload(h.lights, h.numLights);
    f32* d_kernelSigma  = upload(h.kernelSigma, h.kernelRes * h.kernelRes * h.kernelRes);
    f32* d_kernelLength = upload(h.kernelLength, h.kernelRes * h.kernelRes * h.kernelRes);

    core::SceneDevice d_scene;
    d_scene.numSpheres = h.numSpheres;
    d_scene.spheres    = d_spheres;
    d_scene.numMeshes  = h.numMeshes;
    d_scene.meshes     = d_meshes;
    d_scene.vertices       = d_vertices;
    d_scene.faces          = d_faces;
    d_scene.bvhNodes       = d_bvhNodes;
    d_scene.bvhTriIndices  = d_bvhTriIndices;
    d_scene.vertexNormals  = d_vertexNormals;
    d_scene.voxelSignGrid  = d_voxelSignGrid;
    d_scene.voxelBandIndex = d_voxelBandIndex;
    d_scene.voxelBandValue = d_voxelBandValue;
    d_scene.voxelRes       = h.voxelRes;
    d_scene.voxelBandCount = h.voxelBandCount;
    d_scene.lights    = d_lights;
    d_scene.numLights = h.numLights;
    d_scene.kernelSigma  = d_kernelSigma;
    d_scene.kernelLength = d_kernelLength;
    d_scene.kernelRes    = h.kernelRes;
    d_scene.numVertices      = h.numVertices;
    d_scene.numFaces         = h.numFaces;
    d_scene.numBvhNodes      = h.numBvhNodes;
    d_scene.numBvhTriIndices = h.numBvhTriIndices;

    // ——— 2. medium: re-bind mesh-mean / voxel-mean pointers to device memory ———
    core::Medium d_medium = ctx.medium;
    if (d_medium.meanType == core::MeanType::Mesh) {
      d_medium.meshMean.vertices      = d_scene.vertices;
      d_medium.meshMean.faces         = d_scene.faces;
      d_medium.meshMean.bvhNodes      = d_scene.bvhNodes;
      d_medium.meshMean.bvhTriIndices = d_scene.bvhTriIndices;
      d_medium.meshMean.vertexNormals = d_scene.vertexNormals;
    } else if (d_medium.meanType == core::MeanType::Voxel) {
      d_medium.voxelMean.signGrid  = d_scene.voxelSignGrid;
      d_medium.voxelMean.bandIndex = d_scene.voxelBandIndex;
      d_medium.voxelMean.bandValue = d_scene.voxelBandValue;
      d_medium.voxelMean.bandCount = d_scene.voxelBandCount;
      d_medium.voxelMean.res       = d_scene.voxelRes;
    }
    d_medium.kernelField.sigma  = d_scene.kernelSigma;
    d_medium.kernelField.length = d_scene.kernelLength;
    d_medium.kernelField.res    = d_scene.kernelRes;

    // ——— 2b. environment map upload ———
    core::EnvMap d_envMap;
    d_envMap.width  = ctx.envMap.width;
    d_envMap.height = ctx.envMap.height;
    RGB_F32* d_envPixels = upload(ctx.envMap.pixels, u32(ctx.envMap.width * ctx.envMap.height));
    d_envMap.pixels = d_envPixels;

    // ——— 3. output framebuffer (device) ———
    RGB_F32* d_fb = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_fb), fb.bytesize()));

    // ——— 4. launch one thread per pixel ———
    dim3 block(16, 16);
    dim3 grid((u32(w) + block.x - 1) / block.x, (u32(hh) + block.y - 1) / block.y);
    renderKernel<<<grid, block>>>(d_scene, d_medium, ctx.hasMedium, ctx.settings, d_envMap,
                                   cam, spp, d_fb);
    cudaCheckLast();
    cudaCheck(cudaDeviceSynchronize());

    // ——— 5. download output ———
    cudaCheck(cudaMemcpy(fb.pixels, d_fb, fb.bytesize(), cudaMemcpyDeviceToHost));

    // ——— 6. free device memory ———
    cudaFree(d_fb);
    cudaFree(d_spheres);
    cudaFree(d_meshes);
    cudaFree(d_vertices);
    cudaFree(d_faces);
    cudaFree(d_bvhNodes);
    cudaFree(d_bvhTriIndices);
    cudaFree(d_vertexNormals);
    cudaFree(d_voxelSignGrid);
    cudaFree(d_voxelBandIndex);
    cudaFree(d_voxelBandValue);
    cudaFree(d_lights);
    cudaFree(d_kernelSigma);
    cudaFree(d_kernelLength);
    cudaFree(d_envPixels);

    mplogtag("cuda", "render {}x{} spp={} done", w, hh, spp);
  }

}
