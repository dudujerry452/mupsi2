#pragma once

#include <vector>

#include "core/geometry/ray.h"
#include "core/primitive/primitive.h"
#include "core/bvh/bvh.h"
#include "core/tracer/intersect.h"
#include "core/light/light.h"

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
    vec3f*    vertexNormals  = nullptr;  // angle-weighted pseudo-normals (parallel to vertices)

    // Sparse narrow-band SDF mean (used when the GP medium has MeanType::Voxel).
    u8*       voxelSignGrid  = nullptr;  // res^3 bytes
    u32*      voxelBandIndex = nullptr;  // sorted in-band node indices
    f32*      voxelBandValue = nullptr;  // in-band signed distances
    u32 voxelRes       = 0;
    u32 voxelBandCount = 0;

    // Emissive area lights (sampled by NEE; also present as sphere primitives).
    u32        numLights = 0;
    AreaLight* lights    = nullptr;

    // Element counts of the shared arrays above (needed for backend upload).
    u32 numVertices      = 0;
    u32 numFaces         = 0;   // count of u32 entries (3 × triangles)
    u32 numBvhNodes      = 0;
    u32 numBvhTriIndices = 0;
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
    std::vector<vec3f>  vertexNormals;   // angle-weighted pseudo-normals (parallel to vertices)

    // Sparse narrow-band SDF mean (MeanType::Voxel).
    std::vector<u8>  voxelSignGrid;
    std::vector<u32> voxelBandIndex;
    std::vector<f32> voxelBandValue;
    u32 voxelRes = 0;

    // Emissive area lights.
    std::vector<AreaLight> lights;

    void addSphere(const vec3f& center, f32 radius, u16 bsdfId, u16 mediumId, u16 lightId);
    void addLight(const vec3f& center, f32 radius, const RGB_F32& emission);
    // vertices/faces must be world-space.
    void addMesh(const vec3f* verts, u32 nv, const u32* tris, u32 nt, u16 bsdfId, u16 mediumId, u16 lightId);

    // A mesh that is NOT rendered as a hard surface — it only provides a GP
    // mean (MeshMean). Its geometry/BVH still land in the shared arrays.
    std::vector<Mesh> meshMeans;
    void addMeshMean(const vec3f* verts, u32 nv, const u32* tris, u32 nt);
  };

  MPHD HitRecord sceneIntersect(const Ray& ray, const SceneDevice& s);
  MPHD bool      sceneOccluded(const Ray& ray, const SceneDevice& s, u32 excludeLightId = NO_LIGHT);

}
