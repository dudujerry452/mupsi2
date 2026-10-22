#include "core/scene/scene.h"

#include "core/bsdf/bsdf.h"

namespace mps::core {

  void SceneHost::addSphere(const vec3f& center, f32 radius, u16 bsdfId, u16 mediumId, u16 lightId) {    Sphere s;
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

  void SceneHost::addLight(const vec3f& center, f32 radius, const RGB_F32& emission) {
    const u16 lightId = u16(lights.size());
    lights.push_back({center, radius, emission});
    // The light is also a visible/blocking Lambertian sphere primitive.
    addSphere(center, radius, u16(BsdfType::Lambertian), NO_MEDIUM, lightId);
  }

  namespace {

    // Angle-weighted pseudo-normals (Bærentzen & Aanæs 2005). Robust outward
    // direction even for open meshes / locally-flipped faces; used by MeshMean
    // to sign the SDF.
    void computeAngleWeightedNormals(const vec3f* V, const u32* tris, u32 nt, u32 nv, vec3f* outN) {
      for (u32 i = 0; i < nv; ++i) outN[i] = {0.0f, 0.0f, 0.0f};
      for (u32 fi = 0; fi < nt; ++fi) {
        const u32 i0 = tris[3 * fi + 0], i1 = tris[3 * fi + 1], i2 = tris[3 * fi + 2];
        const vec3f a = V[i0], b = V[i1], c = V[i2];
        vec3f n = mps::cross(b - a, c - a);
        const f32 nlen = mps::norm(n);
        if (nlen < 1e-12f) continue;
        n = n * (1.0f / nlen);

        auto angleAt = [&](const vec3f& v, const vec3f& o1, const vec3f& o2) {
          const vec3f e1 = mps::normalize(o1 - v);
          const vec3f e2 = mps::normalize(o2 - v);
          return mps::acosf(mps::clampf(mps::dot(e1, e2), -1.0f, 1.0f));
        };
        outN[i0] = outN[i0] + n * angleAt(a, b, c);
        outN[i1] = outN[i1] + n * angleAt(b, a, c);
        outN[i2] = outN[i2] + n * angleAt(c, a, b);
      }
      for (u32 i = 0; i < nv; ++i) {
        const f32 nl = mps::norm(outN[i]);
        if (nl > 1e-12f) outN[i] = outN[i] * (1.0f / nl);
        else outN[i] = {0.0f, 0.0f, 1.0f};
      }
    }

    // Append world-space triangle soup to the shared arrays and build its BVH.
    // Returns the Mesh metadata (offsets + root). Does not register it anywhere.
    Mesh buildMeshData(SceneHost& h, const vec3f* verts, u32 nv, const u32* tris, u32 nt,
                       u16 bsdfId, u16 mediumId, u16 lightId) {
      Mesh m;
      m.info.type = PrimitiveType::Mesh;
      m.info.bsdfId = bsdfId;
      m.info.texId = 0;
      m.info.mediumId = mediumId;
      m.info.lightId = lightId;
      m.info.wpos = {0.0f, 0.0f, 0.0f};
      m.info.quater = {0.0f, 0.0f, 0.0f, 1.0f};
      m.info.scale = 1.0f;

      m.vertexOffset = u32(h.vertices.size());
      m.vertexCount = nv;

      AABB box = AABB::empty();
      for (u32 i = 0; i < nv; ++i) {
        h.vertices.push_back(verts[i]);
        box.extend(verts[i]);
      }

      m.faceOffset = u32(h.faces.size());
      m.faceCount = nt;
      for (u32 i = 0; i < 3 * nt; ++i)
        h.faces.push_back(tris[i] + m.vertexOffset);  // global vertex indices
      m.info.bounds = box;

      // Angle-weighted pseudo-normals, parallel to h.vertices.
      {
        std::vector<vec3f> nrm(nv);
        computeAngleWeightedNormals(verts, tris, nt, nv, nrm.data());
        h.vertexNormals.insert(h.vertexNormals.end(), nrm.begin(), nrm.end());
      }

      BVH bvh;
      bvh.build(h.vertices.data(), h.faces.data() + m.faceOffset, nt);

      m.bvhRoot = bvh.root();
      m.bvhNodeOffset = u32(h.bvhNodes.size());
      m.bvhNodeCount = u32(bvh.nodes.size());
      m.bvhTriOffset = u32(h.bvhTriIndices.size());
      m.bvhTriCount = u32(bvh.triIndices.size());

      h.bvhNodes.insert(h.bvhNodes.end(), bvh.nodes.begin(), bvh.nodes.end());
      for (u32 ti : bvh.triIndices)
        h.bvhTriIndices.push_back(m.faceOffset / 3 + ti);  // global face index (triangle units)

      return m;
    }

  } // namespace

  void SceneHost::addMesh(const vec3f* verts, u32 nv, const u32* tris, u32 nt,
                          u16 bsdfId, u16 mediumId, u16 lightId) {
    if (nv == 0 || nt == 0) return;
    meshes.push_back(buildMeshData(*this, verts, nv, tris, nt, bsdfId, mediumId, lightId));
  }

  void SceneHost::addMeshMean(const vec3f* verts, u32 nv, const u32* tris, u32 nt) {
    if (nv == 0 || nt == 0) return;
    meshMeans.push_back(buildMeshData(*this, verts, nv, tris, nt, 0, NO_MEDIUM, NO_LIGHT));
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

  MPHD bool sceneOccluded(const Ray& ray, const SceneDevice& s, u32 excludeLightId) {
    for (u32 i = 0; i < s.numSpheres; ++i) {
      const Sphere& sp = s.spheres[i];
      if (excludeLightId != NO_LIGHT && sp.info.lightId == excludeLightId) continue; // the light we target
      if (sp.info.bsdfId == u16(BsdfType::Null)) continue;                            // transparent wrapper
      if (Intersect<PrimitiveType::Sphere>::run(ray, sp).hit) return true;
    }
    MeshGeometry g{s.vertices, s.faces, s.bvhNodes, s.bvhTriIndices};
    for (u32 i = 0; i < s.numMeshes; ++i) {
      const Mesh& m = s.meshes[i];
      if (excludeLightId != NO_LIGHT && m.info.lightId == excludeLightId) continue;
      if (m.info.bsdfId == u16(BsdfType::Null)) continue;
      if (occludedBVH(ray, g.bvhNodes + m.bvhNodeOffset, g.bvhTriIndices + m.bvhTriOffset,
                      g.vertices, g.faces, m.bvhRoot)) return true;
    }
    return false;
  }

}
