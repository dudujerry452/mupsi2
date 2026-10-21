#pragma once

#include <vector>

#include "core/geometry/ray.h"
#include "core/primitive/primitive.h"
#include "core/bvh/bvh.h"
#include "core/tracer/intersect.h"

namespace mps::core {

  // Device-visible scene: everything is POD arrays, no std containers.
  struct SceneDevice {
    u32 numSpheres = 0;
    Sphere* spheres = nullptr;
    u32 numMeshes = 0;
    Mesh* meshes = nullptr;

    vec3f*    vertices       = nullptr;  // all mesh vertices (world space)
    u32*      faces          = nullptr;  // all mesh triangles (3×u32 each)
    BVHNode*  bvhNodes       = nullptr;
    u32*      bvhTriIndices  = nullptr;  // global face indices
  };

  // Host-side staging area (owns its vectors). Built by host::SceneBuilder,
  // then flattened into SceneDevice via host::toDevice.
  struct SceneHost {
    std::vector<Sphere> spheres;
    std::vector<Mesh>   meshes;
    std::vector<vec3f>  vertices;
    std::vector<u32>    faces;
    std::vector<BVHNode> bvhNodes;
    std::vector<u32>     bvhTriIndices;

    void addSphere(const vec3f& center, f32 radius, u16 bsdfId, u16 mediumId, u16 lightId);
    // vertices/faces must be world-space.
    void addMesh(const vec3f* verts, u32 nv, const u32* tris, u32 nt, u16 bsdfId, u16 mediumId, u16 lightId);
  };

  MPHD HitRecord sceneIntersect(const Ray& ray, const SceneDevice& s);
  MPHD bool      sceneOccluded(const Ray& ray, const SceneDevice& s);

}
