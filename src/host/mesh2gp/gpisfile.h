#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "core/basic/types.h"
#include "core/geometry/aabb.h"
#include "core/gp/kernel.h"

// NOTE: mps::vec3f / mps::u32 / mps::f32 live in namespace mps (types.h),
// NOT in mps::core. Qualify accordingly.

// ============================================================================
// GPIS artifact file (.gpibin)
// ============================================================================
//
// Design rule: the file is a **1:1 memory image of the runtime struct**.
// Writing = memcpy the struct's scalars; reading = memcpy straight back.
// No serialization library, no field-by-field encoding, no endian swap.
//
// The struct below (GPISData) is the *portable* form: fixed-width scalars,
// fixed-size arrays inline, everything after the header a flat byte blob
// whose internal offsets are stored as byte offsets (not pointers). Reader
// (mupsi2, or any other host) does one read() into the blob, then rebinds
// pointers via offsets. No per-element parsing anywhere.
//
//   on-disk  ==  in-memory (once pointers are rebound)
//
// Layout:
//   [0, 64)         GPISHeader              (magic, version, section dir)
//   [64, ...)       section blobs, each 64-byte aligned
//
// Every section is a raw POD array. Sections that mupsi2's Medium/SparseSDFGrid
// can consume directly are emitted with the *same* element type, so dropping
// them in is a memcpy — that is the "方便后续移植" requirement.

namespace mps::gpifile {

  constexpr char MAGIC[8] = {'M', 'P', 'G', 'P', 'I', 'S', 0x00, 0x1A};
  // Bump when any section layout changes. Reader rejects a mismatch.
  //   v1: mesh + bvh + pseudo-normal + sparse SDF grid + kernel params
  constexpr u32 VERSION = 1;

  // ——— Section kinds. Order in the file = order of the SectionDesc table ———
  enum class Section : u32 {
    Vertices      = 0,  // vec3f,  numVertices
    Faces         = 1,  // u32×3,  numTriangles
    VertexNormals = 2,  // vec3f,  numVertices  (angle-weighted pseudo-normals)
    BVHNodes      = 3,  // BVHNode as {f32[6] bounds, u32 left, u32 right} × numBvhNodes
    BVHTriIndices = 4,  // u32,    numBvhTriIndices
    SignGrid      = 5,  // u8,     res^3
    BandIndex     = 6,  // u32,    bandCount
    BandValue     = 7,  // f32,    bandCount
    COUNT         = 8
  };

  inline const char* sectionName(Section s) {
    switch (s) {
      case Section::Vertices:      return "Vertices";
      case Section::Faces:         return "Faces";
      case Section::VertexNormals: return "VertexNormals";
      case Section::BVHNodes:      return "BVHNodes";
      case Section::BVHTriIndices: return "BVHTriIndices";
      case Section::SignGrid:      return "SignGrid";
      case Section::BandIndex:     return "BandIndex";
      case Section::BandValue:     return "BandValue";
      default:                     return "?";
    }
  }

  // One section's location inside the blob.
  struct SectionDesc {
    u64 offset = 0;   // byte offset from the start of the blob
    u64 count  = 0;   // element count
    u64 stride = 0;   // bytes per element
  };

  // BVH node as stored on disk: AABB packed as 6 floats, then the same
  // left/right encoding as core::BVHNode (leaf flag in the top bit of right).
  // Explicitly packed so the byte layout can never drift with padding rules.
  struct alignas(4) BVHNodePacked {
    f32 minx, miny, minz;
    f32 maxx, maxy, maxz;
    u32 left;
    u32 right;
  };
  static_assert(sizeof(BVHNodePacked) == 32, "BVHNodePacked must be 32 bytes");

  // The exported kernel parameters — plain scalars, mirrors core::SEKernel plus
  // the sparse-convolution density that generated the field.
  struct KernelParams {
    f32 sigma;          // amplitude
    f32 Lx, Ly, Lz;     // per-axis length scale
    f32 kernelRadius;   // compact support radius (= grid cellSize)
    u32 impulseDensity; // points per cell
    u32 seed;
    u32 pad[9];         // explicit pad: 7 words -> 16 words = 64 bytes, ABI-proof
  };
  static_assert(sizeof(KernelParams) == 64, "KernelParams must be 64 bytes");
  static_assert(alignof(KernelParams) == 4, "KernelParams must be 4-byte aligned");

  // File header. sizeof is a multiple of 64 so the blob starts 64-byte aligned.
  struct alignas(64) Header {
    char magic[8];
    u32  version;
    u32  headerBytes;     // sizeof(Header)
    u64  totalBytes;      // whole file, including header

    // Source mesh stats.
    u32 numVertices;
    u32 numTriangles;
    u32 numBvhNodes;
    u32 numBvhTriIndices;

    // Sparse SDF grid (mupsi2 SparseSDFGrid-compatible).
    f32 gridOriginX, gridOriginY, gridOriginZ;
    f32 gridCellSize;
    f32 gridBand;
    u32 gridRes;
    u32 bandCount;

    // Kernel / noise parameters the field was built with.
    KernelParams kernel;

    // Mesh world AABB (informational; also derivable from Vertices).
    f32 boundsMinX, boundsMinY, boundsMinZ;
    f32 boundsMaxX, boundsMaxY, boundsMaxZ;

    // Section directory: [0, COUNT).
    SectionDesc sections[u32(Section::COUNT)];

    // Byte offset of the first section (= headerBytes, rounded up to 64).
    u64 blobOffset;
    // Uniform scale that was applied to the source mesh before voxelisation.
    // Stored so `check` can rebuild an exact reference mesh without the caller
    // having to remember the build-time --scale (an artifact must describe
    // itself; geometry, SDF lattice and this number are one rigid triple).
    f32 meshScale;
    u32 reserved0;
    u32 pad[10];            // pad to a 64-byte multiple
  };
  static_assert(sizeof(Header) % 64 == 0, "header must pad to a 64-byte multiple");
  static_assert(std::is_trivially_copyable_v<Header>, "Header must be memcpy-able");
  static_assert(std::is_trivially_copyable_v<KernelParams>, "KernelParams must be memcpy-able");

  // ——— Host-side owner of the blob + rebound views ———
  //
  // This is exactly what gets memcpy'd to disk (blob) plus the reconstructed
  // pointers a consumer uses. mupsi2's SparseSDFGrid / MeshMean are filled by
  // bind() without any per-element work.
  struct GPISData {
    Header header{};
    std::vector<u8> blob;   // everything after the header, 64-byte aligned

    // Rebound views into blob (valid after parse()).
    const vec3f*    vertices      = nullptr;
    const u32*      faces         = nullptr;
    const vec3f*    vertexNormals = nullptr;
    const BVHNodePacked*  bvhNodes      = nullptr;
    const u32*      bvhTriIndices = nullptr;
    const u8*       signGrid      = nullptr;
    const u32*      bandIndex     = nullptr;
    const f32*      bandValue     = nullptr;

    // Rebuild the typed views from header + blob. Returns false on a layout
    // inconsistency (offset out of range / stride mismatch).
    bool bind();

    core::AABB bounds() const {
      return {{header.boundsMinX, header.boundsMinY, header.boundsMinZ},
              {header.boundsMaxX, header.boundsMaxY, header.boundsMaxZ}};
    }

    // Total file bytes = sizeof(Header) + blob.size().
    u64 fileBytes() const { return sizeof(Header) + u64(blob.size()); }
  };

  // Write `data` to `path` as one Header + one contiguous blob.
  // The blob is written as-is: it *is* the memory image.
  bool write(const std::string& path, const GPISData& data);

  // Read `path` back into `out`. Verifies magic / version / section bounds.
  bool read(const std::string& path, GPISData& out);

  // Human-readable dump of the section directory (for `mesh2gp --info`).
  std::string describe(const GPISData& data);

} // namespace mps::gpifile
