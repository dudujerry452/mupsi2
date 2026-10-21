#include "host/data_manager.h"

#include <cstring>

namespace mps::host {

  namespace {

    template<typename T>
    T* copyArray(const std::vector<T>& v) {
      if (v.empty()) return nullptr;
      T* p = new T[v.size()];
      std::memcpy(p, v.data(), v.size() * sizeof(T));
      return p;
    }

  } // namespace

  core::SceneDevice toDevice(const core::SceneHost& host) {
    core::SceneDevice dev;
    dev.numSpheres = u32(host.spheres.size());
    dev.numMeshes  = u32(host.meshes.size());

    dev.spheres = copyArray(host.spheres);
    dev.meshes  = copyArray(host.meshes);
    dev.vertices      = copyArray(host.vertices);
    dev.faces         = copyArray(host.faces);
    dev.bvhNodes      = copyArray(host.bvhNodes);
    dev.bvhTriIndices = copyArray(host.bvhTriIndices);
    dev.vertexNormals = copyArray(host.vertexNormals);

    dev.voxelSignGrid  = copyArray(host.voxelSignGrid);
    dev.voxelBandIndex = copyArray(host.voxelBandIndex);
    dev.voxelBandValue = copyArray(host.voxelBandValue);
    dev.voxelRes       = host.voxelRes;
    dev.voxelBandCount = u32(host.voxelBandIndex.size());

    dev.lights    = copyArray(host.lights);
    dev.numLights = u32(host.lights.size());

    dev.kernelSigma  = copyArray(host.kernelSigma);
    dev.kernelLength = copyArray(host.kernelLength);
    dev.kernelRes    = host.kernelRes;

    dev.colorGrid = copyArray(host.colorGrid);
    dev.colorRes  = host.colorRes;

    dev.heatPhi         = copyArray(host.heatPhi);
    dev.heatLam         = copyArray(host.heatLam);
    dev.heatKernelPhi   = copyArray(host.heatKernelPhi);
    dev.heatKernelColor = copyArray(host.heatKernelColor);
    dev.heatKernelTau   = copyArray(host.heatKernelTau);
    dev.heatKernelSharp = copyArray(host.heatKernelSharp);
    dev.heatN = host.heatN; dev.heatK = host.heatK; dev.heatS = host.heatS;
    dev.heatT = host.heatT; dev.heatBhSigma = host.heatBhSigma;
    dev.heatMeanColor = host.heatMeanColor;

    dev.numVertices      = u32(host.vertices.size());
    dev.numFaces         = u32(host.faces.size());
    dev.numBvhNodes      = u32(host.bvhNodes.size());
    dev.numBvhTriIndices = u32(host.bvhTriIndices.size());

    return dev;
  }

  void freeDevice(core::SceneDevice& dev) {
    delete[] dev.spheres;
    delete[] dev.meshes;
    delete[] dev.vertices;
    delete[] dev.faces;
    delete[] dev.bvhNodes;
    delete[] dev.bvhTriIndices;
    delete[] dev.vertexNormals;
    delete[] dev.voxelSignGrid;
    delete[] dev.voxelBandIndex;
    delete[] dev.voxelBandValue;
    delete[] dev.lights;
    delete[] dev.kernelSigma;
    delete[] dev.kernelLength;
    delete[] dev.colorGrid;
    delete[] dev.heatPhi;
    delete[] dev.heatLam;
    delete[] dev.heatKernelPhi;
    delete[] dev.heatKernelColor;
    delete[] dev.heatKernelTau;
    delete[] dev.heatKernelSharp;
    dev = core::SceneDevice{};
  }

  void bindMeshMean(core::Medium& m, const core::SceneDevice& dev, const core::Mesh& ref) {
    m.meshMean.vertices       = dev.vertices;
    m.meshMean.faces          = dev.faces;
    m.meshMean.bvhNodes       = dev.bvhNodes;
    m.meshMean.bvhTriIndices  = dev.bvhTriIndices;
    m.meshMean.vertexNormals  = dev.vertexNormals;
    m.meshMean.bvhRoot        = ref.bvhRoot;
    m.meshMean.bvhNodeOffset  = ref.bvhNodeOffset;
    m.meshMean.bvhTriOffset   = ref.bvhTriOffset;
  }

  void bindVoxelMean(core::Medium& m, const core::SceneDevice& dev) {
    m.voxelMean.signGrid  = dev.voxelSignGrid;
    m.voxelMean.bandIndex = dev.voxelBandIndex;
    m.voxelMean.bandValue = dev.voxelBandValue;
    m.voxelMean.bandCount = dev.voxelBandCount;
    m.voxelMean.res       = dev.voxelRes;
  }

  void bindKernelField(core::Medium& m, const core::SceneDevice& dev) {
    m.kernelField.sigma  = dev.kernelSigma;
    m.kernelField.length = dev.kernelLength;
    m.kernelField.res    = dev.kernelRes;
    // origin / cellSize are set by the scene builder.
  }

  void bindColorGrid(core::Medium& m, const core::SceneDevice& dev) {
    m.colorGrid.data = dev.colorGrid;
    m.colorGrid.res  = dev.colorRes;
    // origin / cellSize are set by the scene builder.
  }

  void bindHeatKernel(core::Medium& m, const core::SceneDevice& dev) {
    m.heatKernel.phi         = dev.heatPhi;
    m.heatKernel.lam         = dev.heatLam;
    m.heatKernel.kernelPhi   = dev.heatKernelPhi;
    m.heatKernel.kernelColor = dev.heatKernelColor;
    m.heatKernel.kernelTau   = dev.heatKernelTau;
    m.heatKernel.kernelSharp = dev.heatKernelSharp;
    m.heatKernel.N = dev.heatN; m.heatKernel.K = dev.heatK; m.heatKernel.S = dev.heatS;
    m.heatKernel.t = dev.heatT; m.heatKernel.bhSigma = dev.heatBhSigma;
    m.heatKernel.meanColor = dev.heatMeanColor;
    // Mesh geometry pointers (vertices/faces/BVH) are bound below from the
    // SceneDevice shared arrays.
    m.heatKernel.vertices      = dev.vertices;
    m.heatKernel.faces         = dev.faces;
    m.heatKernel.bvhNodes      = dev.bvhNodes;
    m.heatKernel.bvhTriIndices = dev.bvhTriIndices;
  }

}
