#include "core/bvh/bvh.h"

#include <algorithm>

namespace mps::core {

  // ——— host-side build (recursion + std::vector are host-only) ———

  namespace {

    AABB triBounds(const vec3f* v, const u32* f, u32 fi) {
      AABB box = AABB::empty();
      box.extend(v[f[3 * fi + 0]]);
      box.extend(v[f[3 * fi + 1]]);
      box.extend(v[f[3 * fi + 2]]);
      return box;
    }

    f32 centroidCoord(const vec3f* v, const u32* f, u32 fi, i32 axis) {
      const vec3f& a = v[f[3 * fi + 0]];
      const vec3f& b = v[f[3 * fi + 1]];
      const vec3f& c = v[f[3 * fi + 2]];
      if (axis == 0) return (a.x + b.x + c.x) / 3.0f;
      if (axis == 1) return (a.y + b.y + c.y) / 3.0f;
      return (a.z + b.z + c.z) / 3.0f;
    }

  } // namespace

  MPHD f32 rayTriIntersect(const Ray& ray, const vec3f& v0, const vec3f& v1, const vec3f& v2,
                           f32& outU, f32& outV) {
    constexpr f32 eps = 1e-8f;

    vec3f e1 = v1 - v0;
    vec3f e2 = v2 - v0;
    vec3f h  = mps::cross(ray.dir, e2);
    f32 a = mps::dot(e1, h);

    if (mps::fabsf(a) < eps) return -1.0f; // parallel

    f32 f = 1.0f / a;
    vec3f s = ray.o - v0;
    f32 u = f * mps::dot(s, h);
    if (u < 0.0f || u > 1.0f) return -1.0f;

    vec3f q = mps::cross(s, e1);
    f32 v = f * mps::dot(ray.dir, q);
    if (v < 0.0f || u + v > 1.0f) return -1.0f;

    f32 t = f * mps::dot(e2, q);
    if (t < ray.min || t > ray.max) return -1.0f;

    outU = u;
    outV = v;
    return t;
  }

  void BVH::build(const vec3f* vertices, const u32* faces, u32 triCount) {
    nodes.clear();
    triIndices.clear();
    if (triCount == 0) return;

    triIndices.resize(triCount);
    for (u32 i = 0; i < triCount; ++i) triIndices[i] = i;
    nodes.reserve(2 * triCount);

    // Recursive lambda over an index range into triIndices.
    auto buildRecursive = [&](auto&& self, u32 begin, u32 end) -> u32 {
      u32 count = end - begin;

      AABB box = AABB::empty();
      for (u32 i = begin; i < end; ++i)
        box = box.unite(triBounds(vertices, faces, triIndices[i]));

      if (count <= 4) {
        u32 idx = u32(nodes.size());
        BVHNode leaf;
        leaf.bounds = box;
        leaf.left   = begin;
        leaf.right  = count | BVHNode::LEAF_FLAG;
        nodes.push_back(leaf);
        return idx;
      }

      i32 axis = box.longestAxis();

      f32 splitMin =  INFINITY;
      f32 splitMax = -INFINITY;
      for (u32 i = begin; i < end; ++i) {
        f32 c = centroidCoord(vertices, faces, triIndices[i], axis);
        splitMin = std::min(splitMin, c);
        splitMax = std::max(splitMax, c);
      }
      f32 splitPos = 0.5f * (splitMin + splitMax);

      auto mid = std::partition(triIndices.begin() + begin, triIndices.begin() + end,
        [&](u32 fi) { return centroidCoord(vertices, faces, fi, axis) < splitPos; });
      u32 midIdx = u32(mid - triIndices.begin());

      if (midIdx == begin || midIdx == end) midIdx = begin + count / 2;

      u32 leftChild  = self(self, begin, midIdx);
      u32 rightChild = self(self, midIdx, end);

      u32 idx = u32(nodes.size());
      BVHNode internal;
      internal.bounds = box;
      internal.left   = leftChild;
      internal.right  = rightChild;
      nodes.push_back(internal);
      return idx;
    };

    buildRecursive(buildRecursive, 0, triCount);
  }

  MPHD f32 intersectBVH(const Ray& ray, const BVHNode* nodes, const u32* triIndices,
                        const vec3f* vertices, const u32* faces, u32 root,
                        u32& outTri, f32& outU, f32& outV) {
    f32 best = ray.max;
    bool found = false;

    u32 stack[64];
    i32 sp = 0;
    stack[sp++] = root;

    while (sp > 0) {
      u32 idx = stack[--sp];
      const BVHNode& node = nodes[idx];

      if (!node.bounds.intersect(ray)) continue;

      if (node.isLeaf()) {
        u32 start = node.leafStart();
        u32 cnt   = node.leafCount();
        for (u32 i = start; i < start + cnt; ++i) {
          u32 fi = triIndices[i];
          const vec3f& v0 = vertices[faces[3 * fi + 0]];
          const vec3f& v1 = vertices[faces[3 * fi + 1]];
          const vec3f& v2 = vertices[faces[3 * fi + 2]];
          f32 u, v;
          f32 t = rayTriIntersect(ray, v0, v1, v2, u, v);
          if (t >= 0.0f && t < best) {
            best = t;
            outTri = fi;
            outU = u;
            outV = v;
            found = true;
          }
        }
      } else {
        u32 left  = node.left;
        u32 right = node.right;
        const BVHNode& ln = nodes[left];
        const BVHNode& rn = nodes[right];
        i32 axis = node.bounds.longestAxis();
        f32 lc, rc, dirAxis;
        if (axis == 0) { lc = ln.bounds.centroid().x; rc = rn.bounds.centroid().x; dirAxis = ray.dir.x; }
        else if (axis == 1) { lc = ln.bounds.centroid().y; rc = rn.bounds.centroid().y; dirAxis = ray.dir.y; }
        else { lc = ln.bounds.centroid().z; rc = rn.bounds.centroid().z; dirAxis = ray.dir.z; }

        // front-to-back: push far child first so near child is popped first
        bool nearFirst = (dirAxis > 0.0f) ? (lc < rc) : (lc > rc);
        u32 first  = nearFirst ? left  : right;
        u32 second = nearFirst ? right : left;
        stack[sp++] = second;
        stack[sp++] = first;
      }
    }

    return found ? best : -1.0f;
  }

  MPHD bool occludedBVH(const Ray& ray, const BVHNode* nodes, const u32* triIndices,
                        const vec3f* vertices, const u32* faces, u32 root) {
    u32 stack[64];
    i32 sp = 0;
    stack[sp++] = root;

    while (sp > 0) {
      u32 idx = stack[--sp];
      const BVHNode& node = nodes[idx];

      if (!node.bounds.intersect(ray)) continue;

      if (node.isLeaf()) {
        u32 start = node.leafStart();
        u32 cnt   = node.leafCount();
        for (u32 i = start; i < start + cnt; ++i) {
          u32 fi = triIndices[i];
          const vec3f& v0 = vertices[faces[3 * fi + 0]];
          const vec3f& v1 = vertices[faces[3 * fi + 1]];
          const vec3f& v2 = vertices[faces[3 * fi + 2]];
          f32 u, v;
          if (rayTriIntersect(ray, v0, v1, v2, u, v) >= 0.0f) return true;
        }
      } else {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    }

    return false;
  }

}
