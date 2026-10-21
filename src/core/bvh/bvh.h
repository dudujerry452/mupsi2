#pragma once

#include <vector>

#include "core/geometry/aabb.h"

namespace mps::core {

  struct BVHNode {
    AABB bounds;
    // Leaf:      left = start index into triIndices, right = (count | LEAF_FLAG)
    // Internal:  left/right = child node indices (no flag)
    u32 left  = 0;
    u32 right = 0;

    static constexpr u32 LEAF_FLAG = 0x80000000u;

    MPHD bool   isLeaf()     const { return (right & LEAF_FLAG) != 0; }
    MPHD u32    leafStart()  const { return left; }
    MPHD u32    leafCount()  const { return right & ~LEAF_FLAG; }
  };

  // Host-side builder over world-space triangle soup (faces = 3×u32 indices).
  struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<u32>     triIndices;  // reordered global face indices

    void build(const vec3f* vertices, const u32* faces, u32 triCount);

    // Root is appended last (post-order build).
    u32 root() const { return nodes.empty() ? 0u : u32(nodes.size() - 1); }
  };

  // Möller–Trumbore ray/triangle. Returns hit t (> 0) or -1 on miss.
  MPHD f32 rayTriIntersect(const Ray& ray, const vec3f& v0, const vec3f& v1, const vec3f& v2,
                           f32& outU, f32& outV);

  // Device traversal (explicit stack, no recursion). Returns nearest hit t or -1.
  MPHD f32 intersectBVH(const Ray& ray, const BVHNode* nodes, const u32* triIndices,
                        const vec3f* vertices, const u32* faces, u32 root,
                        u32& outTri, f32& outU, f32& outV);

  // Early-exit occlusion test.
  MPHD bool occludedBVH(const Ray& ray, const BVHNode* nodes, const u32* triIndices,
                        const vec3f* vertices, const u32* faces, u32 root);

}
