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

    return dev;
  }

  void freeDevice(core::SceneDevice& dev) {
    delete[] dev.spheres;
    delete[] dev.meshes;
    delete[] dev.vertices;
    delete[] dev.faces;
    delete[] dev.bvhNodes;
    delete[] dev.bvhTriIndices;
    dev = core::SceneDevice{};
  }

}
