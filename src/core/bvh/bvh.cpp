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

  MPHD vec3f closestPointOnTriangle(const vec3f& p, const vec3f& a, const vec3f& b, const vec3f& c) {
    const vec3f ab = b - a;
    const vec3f ac = c - a;
    const vec3f ap = p - a;
    const f32 d1 = mps::dot(ab, ap);
    const f32 d2 = mps::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    const vec3f bp = p - b;
    const f32 d3 = mps::dot(ab, bp);
    const f32 d4 = mps::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    const f32 vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
      const f32 v = d1 / (d1 - d3);
      return a + ab * v;
    }

    const vec3f cp = p - c;
    const f32 d5 = mps::dot(ab, cp);
    const f32 d6 = mps::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    const f32 vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
      const f32 w = d2 / (d2 - d6);
      return a + ac * w;
    }

    const f32 va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
      const f32 w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      return b + (c - b) * w;
    }

    const f32 denom = 1.0f / (va + vb + vc);
    const f32 v = vb * denom;
    const f32 w = vc * denom;
    return a + ab * v + ac * w;
  }

  MPHD f32 closestPointBVH(const vec3f& p, const BVHNode* nodes, const u32* triIndices,
                           const vec3f* vertices, const u32* faces, u32 root,
                           u32& outTri, vec3f& outPoint) {
    f32 bestDist2 = INFINITY;
    bool found = false;
    u32 stack[64];
    i32 sp = 0;
    stack[sp++] = root;

    while (sp > 0) {
      const u32 idx = stack[--sp];
      const BVHNode& node = nodes[idx];

      // Squared distance from p to the node's AABB (0 if inside).
      const vec3f q = mps::clampv(p, node.bounds.min, node.bounds.max);
      if (mps::norm2(p - q) >= bestDist2) continue;

      if (node.isLeaf()) {
        const u32 start = node.leafStart();
        const u32 cnt   = node.leafCount();
        for (u32 i = start; i < start + cnt; ++i) {
          const u32 fi = triIndices[i];
          const vec3f& v0 = vertices[faces[3 * fi + 0]];
          const vec3f& v1 = vertices[faces[3 * fi + 1]];
          const vec3f& v2 = vertices[faces[3 * fi + 2]];
          const vec3f c = closestPointOnTriangle(p, v0, v1, v2);
          const f32 d2 = mps::norm2(p - c);
          if (d2 < bestDist2) {
            bestDist2 = d2;
            outTri = fi;
            outPoint = c;
            found = true;
          }
        }
      } else {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    }

    return found ? mps::sqrtf(bestDist2) : INFINITY;
  }

  MPHD i32 countIntersectionsBVH(const Ray& ray, const BVHNode* nodes, const u32* triIndices,
                                 const vec3f* vertices, const u32* faces, u32 root) {
    i32 count = 0;
    u32 stack[64];
    i32 sp = 0;
    stack[sp++] = root;

    while (sp > 0) {
      const u32 idx = stack[--sp];
      const BVHNode& node = nodes[idx];

      if (!node.bounds.intersect(ray)) continue;

      if (node.isLeaf()) {
        const u32 start = node.leafStart();
        const u32 cnt   = node.leafCount();
        for (u32 i = start; i < start + cnt; ++i) {
          const u32 fi = triIndices[i];
          const vec3f& v0 = vertices[faces[3 * fi + 0]];
          const vec3f& v1 = vertices[faces[3 * fi + 1]];
          const vec3f& v2 = vertices[faces[3 * fi + 2]];
          f32 u, v;
          if (rayTriIntersect(ray, v0, v1, v2, u, v) >= 0.0f) ++count;
        }
      } else {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    }

    return count;
  }

}
