#include "host/mesh2gp/gpisfile.h"

#include <cstdio>
#include <cstring>
#include <format>
#include <fstream>

#include "core/utils/log.h"

namespace mps::gpifile {

  namespace {

    constexpr u64 alignUp(u64 v, u64 a) { return (v + a - 1) / a * a; }

    // Section element size, by kind. Kept in one place so writer and reader
    // can never disagree.
    u64 elementSize(Section s) {
      switch (s) {
        case Section::Vertices:      return sizeof(mps::vec3f);
        case Section::Faces:         return sizeof(mps::u32);
        case Section::VertexNormals: return sizeof(mps::vec3f);
        case Section::BVHNodes:      return sizeof(BVHNodePacked);
        case Section::BVHTriIndices: return sizeof(mps::u32);
        case Section::SignGrid:      return sizeof(mps::u8);
        case Section::BandIndex:     return sizeof(mps::u32);
        case Section::BandValue:     return sizeof(mps::f32);
        default:                     return 0;
      }
    }

    // Number of elements the header says a section contains.
    u64 expectedCount(const Header& h, Section s) {
      switch (s) {
        case Section::Vertices:      return h.numVertices;
        case Section::Faces:         return u64(h.numTriangles) * 3;
        case Section::VertexNormals: return h.numVertices;
        case Section::BVHNodes:      return h.numBvhNodes;
        case Section::BVHTriIndices: return h.numBvhTriIndices;
        case Section::SignGrid:      return u64(h.gridRes) * h.gridRes * h.gridRes;
        case Section::BandIndex:     return h.bandCount;
        case Section::BandValue:     return h.bandCount;
        default:                     return 0;
      }
    }

  } // namespace

  bool GPISData::bind() {
    vertices = nullptr; faces = nullptr; vertexNormals = nullptr;
    bvhNodes = nullptr; bvhTriIndices = nullptr;
    signGrid = nullptr; GPISData::bandIndex = nullptr; GPISData::bandValue = nullptr;

    const u8* base = blob.data();
    const u64 blobBytes = blob.size();

    for (u32 i = 0; i < u32(Section::COUNT); ++i) {
      const Section s = Section(i);
      const SectionDesc& d = header.sections[i];
      if (d.count == 0) continue;

      // Every descriptor must describe a region fully inside the blob, with the
      // stride and count the header implies for that section.
      if (d.stride != elementSize(s)) {
        mplogtag("gpifile", "bind: section {} stride {} != expected {}", i, d.stride, elementSize(s));
        return false;
      }
      if (d.count != expectedCount(header, s)) {
        mplogtag("gpifile", "bind: section {} count {} != expected {}", i, d.count, expectedCount(header, s));
        return false;
      }
      const u64 end = d.offset + d.count * d.stride;
      if (end > blobBytes) {
        mplogtag("gpifile", "bind: section {} overruns blob ({} > {})", i, end, blobBytes);
        return false;
      }

      // Pointer arithmetic only — no copying, no per-element loop.
      // NOTE: member names shadow the enum values here, so write each member
      // explicitly instead of relying on unqualified lookup.
      const mps::u8* p = base + d.offset;
      switch (s) {
        case Section::Vertices:      this->vertices      = reinterpret_cast<const mps::vec3f*>(p); break;
        case Section::Faces:         this->faces         = reinterpret_cast<const mps::u32*>(p);   break;
        case Section::VertexNormals: this->vertexNormals = reinterpret_cast<const mps::vec3f*>(p); break;
        case Section::BVHNodes:      this->bvhNodes      = reinterpret_cast<const BVHNodePacked*>(p); break;
        case Section::BVHTriIndices: this->bvhTriIndices = reinterpret_cast<const mps::u32*>(p);   break;
        case Section::SignGrid:      this->signGrid      = reinterpret_cast<const mps::u8*>(p);    break;
        case Section::BandIndex:     this->bandIndex     = reinterpret_cast<const mps::u32*>(p);   break;
        case Section::BandValue:     this->bandValue     = reinterpret_cast<const mps::f32*>(p);   break;
        default: break;
      }
    }
    return true;
  }

  bool write(const std::string& path, const GPISData& data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
      mplogtag("gpifile", "write: cannot open {}", path);
      return false;
    }
    // Header and blob are separate write()s, but the byte stream is identical to
    // one contiguous memory image: header, then blob as-is.
    f.write(reinterpret_cast<const char*>(&data.header), sizeof(Header));
    if (!data.blob.empty())
      f.write(reinterpret_cast<const char*>(data.blob.data()), std::streamsize(data.blob.size()));
    if (!f) {
      mplogtag("gpifile", "write: failed while writing {}", path);
      return false;
    }
    mplogtag("gpifile", "wrote {} ({} bytes: header {} + blob {})",
             path, data.fileBytes(), sizeof(Header), data.blob.size());
    return true;
  }

  bool read(const std::string& path, GPISData& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      mplogtag("gpifile", "read: cannot open {}", path);
      return false;
    }

    if (f.read(reinterpret_cast<char*>(&out.header), sizeof(Header)); !f) {
      mplogtag("gpifile", "read: truncated header");
      return false;
    }
    if (std::memcmp(out.header.magic, MAGIC, sizeof(MAGIC)) != 0) {
      mplogtag("gpifile", "read: bad magic (not a .gpibin)");
      return false;
    }
    if (out.header.version != VERSION) {
      mplogtag("gpifile", "read: version {} != supported {}", out.header.version, VERSION);
      return false;
    }
    if (out.header.totalBytes < sizeof(Header)) {
      mplogtag("gpifile", "read: totalBytes {} < header size", out.header.totalBytes);
      return false;
    }

    const u64 blobBytes = out.header.totalBytes - sizeof(Header);
    out.blob.resize(size_t(blobBytes));
    if (blobBytes > 0) {
      f.read(reinterpret_cast<char*>(out.blob.data()), std::streamsize(blobBytes));
      if (!blobBytes || f.gcount() != std::streamsize(blobBytes)) {
        mplogtag("gpifile", "read: truncated blob");
        return false;
      }
    }
    return out.bind();
  }

  std::string describe(const GPISData& data) {
    const Header& h = data.header;
    std::string s;
    s += std::format("gpibin v{}  total {} bytes (header {} + blob {})\n",
                     h.version, h.totalBytes, sizeof(Header), data.blob.size());
    s += std::format("  mesh: {} verts, {} tris, {} bvh nodes, {} bvh tris\n",
                     h.numVertices, h.numTriangles, h.numBvhNodes, h.numBvhTriIndices);
    s += std::format("  grid: res {}  origin ({:.4f}, {:.4f}, {:.4f})  cell {:.6f}  band {:.6f}\n",
                     h.gridRes, h.gridOriginX, h.gridOriginY, h.gridOriginZ,
                     h.gridCellSize, h.gridBand);
    s += std::format("  band: {} / {} nodes in narrow band ({:.2f}%)\n",
                     h.bandCount, u64(h.gridRes) * h.gridRes * h.gridRes,
                     h.gridRes ? 100.0 * double(h.bandCount) /
                                     double(u64(h.gridRes) * h.gridRes * h.gridRes) : 0.0);
    s += std::format("  kernel: sigma {:.4f}  L ({:.4f}, {:.4f}, {:.4f})  radius {:.4f}  density {}  seed {}\n",
                     h.kernel.sigma, h.kernel.Lx, h.kernel.Ly, h.kernel.Lz,
                     h.kernel.kernelRadius, h.kernel.impulseDensity, h.kernel.seed);
    s += std::format("  bounds: ({:.4f}, {:.4f}, {:.4f}) .. ({:.4f}, {:.4f}, {:.4f})\n",
                     h.boundsMinX, h.boundsMinY, h.boundsMinZ,
                     h.boundsMaxX, h.boundsMaxY, h.boundsMaxZ);
    s += "  sections:\n";
    for (u32 i = 0; i < u32(Section::COUNT); ++i) {
      const SectionDesc& d = h.sections[i];
      s += std::format("    {:>2} {:<14} offset {:>10}  count {:>10}  stride {:>3}  bytes {:>12}\n",
                       i, sectionName(Section(i)), d.offset, d.count, d.stride,
                       d.count * d.stride);
    }
    return s;
  }

} // namespace mps::gpifile
