#include "host/mesh2gp/generator.h"

#include <cstring>
#include <vector>

#include "core/scene/scene.h"
#include "core/utils/log.h"
#include "host/obj_loader.h"
#include "host/voxel_builder.h"

namespace mps::host {

  namespace {

    // Append `nbytes` of `src` to `blob`, 64-byte aligned, and record the
    // section descriptor. Returns the offset the section landed at.
    u64 appendSection(gpifile::GPISData& out, gpifile::Section s,
                      const void* src, u64 count, u64 stride) {
      const u64 pad = (64 - (out.blob.size() % 64)) % 64;
      out.blob.insert(out.blob.end(), size_t(pad), u8(0));

      const u64 offset = u64(out.blob.size());
      const u64 bytes = count * stride;
      if (bytes > 0) {
        const u8* p = static_cast<const u8*>(src);
        out.blob.insert(out.blob.end(), p, p + bytes);
      }

      gpifile::SectionDesc& d = out.header.sections[u32(s)];
      d.offset = offset;
      d.count  = count;
      d.stride = stride;
      return offset;
    }

  } // namespace

  bool buildGpisFromObj(const std::string& objPath, const GpisBuildParams& p,
                        gpifile::GPISData& out) {
    // ——— 1. Load the OBJ ———
    ObjMesh obj;
    if (!loadObj(objPath, obj)) {
      mplog("buildGpisFromObj: failed to load {}", objPath);
      return false;
    }
    mplog("loaded {}: {} verts, {} tris", objPath,
          obj.vertices.size(), obj.faces.size() / 3);

    if (obj.faces.empty()) {
      mplog("buildGpisFromObj: no triangles in {}", objPath);
      return false;
    }

    // ——— 2. Apply uniform scale ———
    std::vector<vec3f> verts = obj.vertices;
    if (p.scale != 1.0f)
      for (vec3f& v : verts) v = v * p.scale;

    const u32 nv = u32(verts.size());
    const u32 nt = u32(obj.faces.size() / 3);

    // ——— 3. Reuse mupsi2's mesh pipeline verbatim ———
    // SceneHost::addMesh computes angle-weighted pseudo-normals and builds the
    // BVH, and stores world-space geometry in exactly the arrays SceneDevice
    // exposes. Nothing new is invented here.
    core::SceneHost host;
    host.addMesh(verts.data(), nv, obj.faces.data(), nt,
                 u16(0), core::NO_MEDIUM, core::NO_LIGHT);

    if (host.meshes.empty()) {
      mplog("buildGpisFromObj: addMesh produced no mesh");
      return false;
    }
    const core::Mesh& mesh = host.meshes[0];
    const core::AABB bounds = mesh.info.bounds;
    if (!bounds.isValid()) {
      mplog("buildGpisFromObj: mesh AABB invalid");
      return false;
    }

    // ——— 4. Sparse narrow-band SDF grid (mupsi2's own builder) ———
    core::MeshMean mm;
    mm.vertices      = host.vertices.data();
    mm.faces         = host.faces.data();
    mm.bvhNodes      = host.bvhNodes.data();
    mm.bvhTriIndices = host.bvhTriIndices.data();
    mm.vertexNormals = host.vertexNormals.data();
    mm.bvhRoot       = mesh.bvhRoot;
    mm.bvhNodeOffset = mesh.bvhNodeOffset;
    mm.bvhTriOffset  = mesh.bvhTriOffset;

    const VoxelGridHost grid = buildSparseSDFGrid(mm, bounds, p.gridRes, p.bandCells);
    if (grid.res == 0 || grid.signGrid.empty()) {
      mplog("buildGpisFromObj: SDF grid build failed");
      return false;
    }

    // ——— 5. Fill the header ———
    out = gpifile::GPISData{};
    gpifile::Header& h = out.header;
    std::memcpy(h.magic, gpifile::MAGIC, sizeof(gpifile::MAGIC));
    h.version     = gpifile::VERSION;
    h.headerBytes = u32(sizeof(gpifile::Header));

    h.numVertices      = nv;
    h.numTriangles     = nt;
    h.numBvhNodes      = u32(host.bvhNodes.size());
    h.numBvhTriIndices = u32(host.bvhTriIndices.size());
    // Self-description: the scale baked into this artifact. `check` rebuilds its
    // reference mesh from the raw OBJ using this, so the two never disagree.
    h.meshScale = p.scale;

    h.gridOriginX = grid.origin.x;
    h.gridOriginY = grid.origin.y;
    h.gridOriginZ = grid.origin.z;
    h.gridCellSize = grid.cellSize;
    h.gridBand     = grid.band;
    h.gridRes      = grid.res;
    h.bandCount    = u32(grid.bandIndex.size());

    h.kernel.sigma = p.sigma;
    h.kernel.Lx = p.Lx;
    h.kernel.Ly = p.Ly;
    h.kernel.Lz = p.Lz;
    h.kernel.kernelRadius = (p.kernelRadius > 0.0f)
                                ? p.kernelRadius
                                : mps::maxf(p.Lx, mps::maxf(p.Ly, p.Lz));
    h.kernel.impulseDensity = p.impulseDensity;
    h.kernel.seed = p.seed;

    h.boundsMinX = bounds.min.x; h.boundsMinY = bounds.min.y; h.boundsMinZ = bounds.min.z;
    h.boundsMaxX = bounds.max.x; h.boundsMaxY = bounds.max.y; h.boundsMaxZ = bounds.max.z;

    // ——— 6. Pack the blob: one aligned append per section, no encoding ———
    out.blob.clear();
    out.blob.reserve(size_t(nv) * sizeof(vec3f) * 2 +
                     size_t(nt) * 3 * sizeof(u32) +
                     size_t(host.bvhNodes.size()) * sizeof(gpifile::BVHNodePacked) +
                     size_t(host.bvhTriIndices.size()) * sizeof(u32) +
                     size_t(grid.signGrid.size()) +
                     size_t(grid.bandIndex.size()) * (sizeof(u32) + sizeof(f32)) +
                     size_t(grid.res) * 64);

    appendSection(out, gpifile::Section::Vertices,
                  host.vertices.data(), nv, sizeof(vec3f));
    appendSection(out, gpifile::Section::Faces,
                  host.faces.data(), u64(nt) * 3, sizeof(u32));
    appendSection(out, gpifile::Section::VertexNormals,
                  host.vertexNormals.data(), nv, sizeof(vec3f));

    // BVH nodes: repack AABB{min,max} + left/right into the fixed 32-byte form.
    {
      std::vector<gpifile::BVHNodePacked> packed(host.bvhNodes.size());
      for (size_t i = 0; i < host.bvhNodes.size(); ++i) {
        const core::BVHNode& n = host.bvhNodes[i];
        gpifile::BVHNodePacked& o = packed[i];
        o.minx = n.bounds.min.x; o.miny = n.bounds.min.y; o.minz = n.bounds.min.z;
        o.maxx = n.bounds.max.x; o.maxy = n.bounds.max.y; o.maxz = n.bounds.max.z;
        o.left  = n.left;
        o.right = n.right;
      }
      appendSection(out, gpifile::Section::BVHNodes,
                    packed.data(), packed.size(), sizeof(gpifile::BVHNodePacked));
    }

    appendSection(out, gpifile::Section::BVHTriIndices,
                  host.bvhTriIndices.data(), host.bvhTriIndices.size(), sizeof(u32));

    appendSection(out, gpifile::Section::SignGrid,
                  grid.signGrid.data(), grid.signGrid.size(), sizeof(u8));
    appendSection(out, gpifile::Section::BandIndex,
                  grid.bandIndex.data(), grid.bandIndex.size(), sizeof(u32));
    appendSection(out, gpifile::Section::BandValue,
                  grid.bandValue.data(), grid.bandValue.size(), sizeof(f32));

    // Blob is 64-aligned by construction, so the first section sits at 0 and
    // the whole file stays aligned without extra padding between header and blob.
    h.blobOffset = 0;
    h.totalBytes = u64(sizeof(gpifile::Header)) + u64(out.blob.size());

    // ——— 7. Rebind views and self-check ———
    if (!out.bind()) {
      mplog("buildGpisFromObj: bind() self-check failed");
      return false;
    }

    mplog("built GPIS: {} verts, {} tris, grid {}^3 ({:.1f}% band), {} bytes",
          nv, nt, grid.res,
          100.0 * double(grid.bandIndex.size()) /
              double(u64(grid.res) * grid.res * grid.res),
          h.totalBytes);
    return true;
  }

} // namespace mps::host
