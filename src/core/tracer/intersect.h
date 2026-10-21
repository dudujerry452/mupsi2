#pragma once

#include "core/geometry/ray.h"
#include "core/primitive/primitive.h"
#include "core/bvh/bvh.h"

namespace mps::core {

  struct HitRecord {
    bool hit = false;

    f32   t = 0.0f;    // fly distance along ray
    vec3f p;           // world hit point
    vec3f wo;          // where the ray came from (=-ray.dir)

    vec3f ng;          // geometric normal (points against the ray)
    vec3f tan, bitan;  // tangent frame

    vec2f uv;          // surface uv
    u16 texId   = 0;
    u16 bsdfId  = 0;
    u16 mediumId = NO_MEDIUM;  // medium wrapped inside the hit primitive
    u16 lightId = NO_LIGHT;
  };

  // Shared device arrays for all meshes.
  struct MeshGeometry {
    const vec3f*   vertices;
    const u32*     faces;         // 3×u32 per triangle
    const BVHNode* bvhNodes;
    const u32*     bvhTriIndices; // global face indices
  };

  template<PrimitiveType T> struct Intersect;

  template<> struct Intersect<PrimitiveType::Sphere> {
    MPHD static HitRecord run(const Ray& ray, const Sphere& s);
  };

  template<> struct Intersect<PrimitiveType::Mesh> {
    MPHD static HitRecord run(const Ray& ray, const Mesh& m, const MeshGeometry& g);
  };

}
