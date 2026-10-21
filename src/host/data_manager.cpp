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

}
