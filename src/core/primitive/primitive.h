#pragma once

#include "core/basic/core.h"
#include "core/geometry/aabb.h"

namespace mps::core {

  enum class PrimitiveType : u16 { Sphere = 0, Mesh = 1 };

  constexpr u16 NO_MEDIUM = 0xFFFFu;
  constexpr u16 NO_LIGHT  = 0xFFFFu;

  // Common header shared by every primitive. Sphere uses info.scale as its
  // radius and info.wpos as its center; Mesh stores pre-transformed world-space
  // geometry in SceneDevice and uses info only for metadata + bounds.
  struct alignas(16) PrimitiveInfo {
    PrimitiveType type;
    u16 bsdfId;    // BsdfType
    u16 texId;     // Texture id
    u16 mediumId;  // NO_MEDIUM = none, else the wrapped medium
    u16 lightId;   // NO_LIGHT = not a light
    vec3f wpos;
    vec4f quater;
    f32  scale;
    AABB bounds;
  };

  struct Sphere {
    PrimitiveInfo info;   // radius = info.scale, center = info.wpos
  };

  struct Mesh {
    PrimitiveInfo info;
    u32 vertexOffset;   // into SceneDevice::vertices
    u32 vertexCount;
    u32 faceOffset;     // into SceneDevice::faces (3×u32 per triangle)
    u32 faceCount;
    u32 bvhRoot;        // root node index into SceneDevice::bvhNodes
    u32 bvhNodeOffset;
    u32 bvhNodeCount;
    u32 bvhTriOffset;   // into SceneDevice::bvhTriIndices
    u32 bvhTriCount;
  };

}
