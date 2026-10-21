#include "core/scene/scene.h"

namespace mps::core {

  void SceneHost::addSphere(const vec3f& center, f32 radius, u16 bsdfId, u16 mediumId, u16 lightId) {
    Sphere s;
    s.info.type = PrimitiveType::Sphere;
    s.info.bsdfId = bsdfId;
    s.info.texId = 0;
    s.info.mediumId = mediumId;
    s.info.lightId = lightId;
    s.info.wpos = center;
    s.info.quater = {0.0f, 0.0f, 0.0f, 1.0f};
    s.info.scale = radius;
    s.info.bounds = AABB(center - vec3f{radius, radius, radius},
                         center + vec3f{radius, radius, radius});
    spheres.push_back(s);
  }

  void SceneHost::addMesh(const vec3f* verts, u32 nv, const u32* tris, u32 nt,
                          u16 bsdfId, u16 mediumId, u16 lightId) {
    if (nv == 0 || nt == 0) return;

    Mesh m;
    m.info.type = PrimitiveType::Mesh;
    m.info.bsdfId = bsdfId;
    m.info.texId = 0;
    m.info.mediumId = mediumId;
    m.info.lightId = lightId;
    m.info.wpos = {0.0f, 0.0f, 0.0f};
    m.info.quater = {0.0f, 0.0f, 0.0f, 1.0f};
    m.info.scale = 1.0f;

    m.vertexOffset = u32(vertices.size());
    m.vertexCount = nv;

    AABB box = AABB::empty();
    for (u32 i = 0; i < nv; ++i) {
      vertices.push_back(verts[i]);
      box.extend(verts[i]);
    }

    m.faceOffset = u32(faces.size());
    m.faceCount = nt;
    for (u32 i = 0; i < 3 * nt; ++i)
      faces.push_back(tris[i] + m.vertexOffset);  // global vertex indices
    m.info.bounds = box;

    // Build this mesh's own BVH over the global faces/vertices arrays.
    BVH bvh;
    bvh.build(vertices.data(), faces.data() + m.faceOffset, nt);

    m.bvhRoot = bvh.root();
    m.bvhNodeOffset = u32(bvhNodes.size());
    m.bvhNodeCount = u32(bvh.nodes.size());
    m.bvhTriOffset = u32(bvhTriIndices.size());
    m.bvhTriCount = u32(bvh.triIndices.size());

    bvhNodes.insert(bvhNodes.end(), bvh.nodes.begin(), bvh.nodes.end());
    for (u32 ti : bvh.triIndices)
      bvhTriIndices.push_back(m.faceOffset + ti);  // global face indices

    meshes.push_back(m);
  }

  MPHD HitRecord sceneIntersect(const Ray& ray, const SceneDevice& s) {
    HitRecord best;
    best.hit = false;
    best.t = ray.max;

    for (u32 i = 0; i < s.numSpheres; ++i) {
      HitRecord h = Intersect<PrimitiveType::Sphere>::run(ray, s.spheres[i]);
      if (h.hit && h.t < best.t) best = h;
    }

    MeshGeometry g{s.vertices, s.faces, s.bvhNodes, s.bvhTriIndices};
    for (u32 i = 0; i < s.numMeshes; ++i) {
      HitRecord h = Intersect<PrimitiveType::Mesh>::run(ray, s.meshes[i], g);
      if (h.hit && h.t < best.t) best = h;
    }

    return best;
  }

  MPHD bool sceneOccluded(const Ray& ray, const SceneDevice& s) {
    for (u32 i = 0; i < s.numSpheres; ++i) {
      if (Intersect<PrimitiveType::Sphere>::run(ray, s.spheres[i]).hit) return true;
    }
    MeshGeometry g{s.vertices, s.faces, s.bvhNodes, s.bvhTriIndices};
    for (u32 i = 0; i < s.numMeshes; ++i) {
      const Mesh& m = s.meshes[i];
      if (occludedBVH(ray, g.bvhNodes + m.bvhNodeOffset, g.bvhTriIndices + m.bvhTriOffset,
                      g.vertices, g.faces, m.bvhRoot)) return true;
    }
    return false;
  }

}
