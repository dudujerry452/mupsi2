#include "host/mesh2gp/scene_compare.h"

#include <vector>

#include "core/bsdf/bsdf.h"
#include "core/utils/log.h"
#include "host/mesh2gp/gpisfile.h"
#include "host/obj_loader.h"

namespace mps::host::m2g {

  namespace {
    // Same auto-fit convention as scene_builder: centre the mesh, scale its
    // diagonal to 2*radius, then drop it onto the floor (y = 0).
    std::vector<vec3f> fitToSphere(const std::vector<vec3f>& src, f32 radius, const vec3f& target) {
      core::AABB box = core::AABB::empty();
      for (const vec3f& v : src) box.extend(v);
      const vec3f center = box.centroid();
      const f32 diagLen = mps::norm(box.diagonal());
      const f32 scale = (diagLen > 0.0f) ? (2.0f * radius / diagLen) : 1.0f;

      std::vector<vec3f> out(src.size());
      for (size_t i = 0; i < src.size(); ++i)
        out[i] = (src[i] - center) * scale + target;

      f32 minY = INFINITY;
      for (const vec3f& v : out) minY = mps::minf(minY, v.y);
      for (vec3f& v : out) v.y -= minY;
      return out;
    }
  } // namespace

  CompareConfig buildCompareScene(const std::string& objPath,
                                  const gpifile::GPISData& art,
                                  const CompareParams& p) {
    CompareConfig cfg;
    cfg.settings.maxBounce = 10;
    cfg.settings.eps      = core::EPS_SURFACE;
    cfg.settings.nearEps  = core::EPS_NEAR;
    cfg.spp = 1;

    ObjMesh obj;
    if (!loadObj(objPath, obj)) {
      mplog("buildCompareScene: cannot load {}", objPath);
      return cfg;
    }

    // ——— Layout (main-renderer units: object radius 150, camera at origin) ———
    const f32 R  = p.objectRadius;
    const f32 dx = 0.5f * p.gap * R;
    const f32 zObj = -(p.camDist * R);          // objects sit in front of the camera
    const vec3f leftTarget {-dx, 0.0f, zObj};
    const vec3f rightTarget{ dx, 0.0f, zObj};

    // ——— LEFT: plain triangle mesh (hard surface, no medium) ———
    {
      std::vector<vec3f> v = fitToSphere(obj.vertices, R, leftTarget);
      cfg.host.addMesh(v.data(), u32(v.size()), obj.faces.data(), u32(obj.faces.size() / 3),
                       u16(core::BsdfType::Lambertian), core::NO_MEDIUM, core::NO_LIGHT);
    }

    // ——— RIGHT: the same object as a GPIS medium ———
    // The artifact holds a complete sparse SDF grid (origin, cellSize, band, res)
    // plus its sign/band arrays. A GPIS is placed rigidly and a translation keeps
    // the voxel lattice axis-aligned, so the grid is reused verbatim — only its
    // origin moves. The artifact must therefore ALREADY be at the drawing size
    // (bake it with --scale); rescaling the lattice without rescaling the
    // geometry inside it would desynchronise field and geometry.
    {
      const core::AABB artBox = art.bounds();
      const vec3f artCentre   = artBox.centroid();
      const f32   artRadius   = 0.5f * mps::norm(artBox.diagonal());
      const f32   unitScale   = (artRadius > 1e-9f) ? (R / artRadius) : 1.0f;

      if (mps::fabsf(unitScale - 1.0f) > 1e-3f) {
        mplog("buildCompareScene: ERROR artifact radius {:.5f} != objectRadius {:.4f}. "
              "Bake the artifact at this size (--scale) so the lattice stays 1:1 "
              "with the geometry it samples.", artRadius, R);
        cfg.hasMedium = false;
        return cfg;
      }

      // Move the artifact onto the right-hand slot and rest it on the floor.
      const vec3f offset{ rightTarget.x - artCentre.x,
                          -artBox.min.y,
                          rightTarget.z - artCentre.z };

      core::Medium& m = cfg.medium;
      m.type     = core::MediumType::GP;
      m.meanType = core::MeanType::Voxel;

      const gpifile::Header& H = art.header;
      m.voxelMean.origin   = {H.gridOriginX + offset.x,
                              H.gridOriginY + offset.y,
                              H.gridOriginZ + offset.z};
      m.voxelMean.cellSize = H.gridCellSize;
      m.voxelMean.band     = H.gridBand;
      m.voxelMean.res      = H.gridRes;

      const gpifile::KernelParams& kp = H.kernel;
      m.sigma          = kp.sigma;
      m.lengthScale    = {kp.Lx, kp.Ly, kp.Lz};
      m.kernelRadius   = kp.kernelRadius;
      m.impulseDensity = kp.impulseDensity;
      m.gpMode         = core::GPMode::SingleRealization;
      m.surfaceBsdf    = u16(core::BsdfType::Lambertian);
      m.colorMethod    = u16(core::ColorMethod::Constant);

      cfg.medium.sceneScale = R;

      const f32 ext = f32(H.gridRes) * m.voxelMean.cellSize;
      cfg.medium.boundMin = m.voxelMean.origin;
      cfg.medium.boundMax = m.voxelMean.origin + vec3f{ext, ext, ext};
      cfg.hasMedium = true;
    }

    // ——— Lights: exactly scene_builder's bunny/spot rig ———
    // ——— Lights: scene_builder's bunny/spot rig, anchored relative to the
    // object plane (zObj) instead of absolute world z.
    // In the original buildMeshGpScene the objects sit at z = -500 and the rig
    // is key=(-250,350,-350), fill=(300,80,-200) — i.e. 150 and 300 units in
    // FRONT of the objects (closer to the camera). Hard-coding z=-350/-200 here
    // silently became "behind the objects" as soon as camDist changed, which
    // put everything in shadow. Keep the rig's relative placement.
    cfg.host.addLight({-250.0f, 350.0f, zObj + 150.0f}, 70.0f, {26.0f, 23.0f, 19.0f});  // key
    cfg.host.addLight({ 300.0f,  80.0f, zObj + 300.0f}, 45.0f, {10.0f,  8.0f,  7.0f});  // fill

    // ——— Floor + back wall (scene_builder's addFloorWalls) ———
    if (p.floor) {
      const f32 floorY = -300.0f, backZ = -850.0f, extent = 500.0f;
      const f32 zNear = zObj + 200.0f;
      const std::vector<vec3f> fv = {{-extent, floorY, zNear}, {extent, floorY, zNear},
                                     {extent, floorY, backZ}, {-extent, floorY, backZ}};
      const std::vector<u32>   ff = {0, 1, 2, 0, 2, 3};
      cfg.host.addMesh(fv.data(), 4, ff.data(), 2,
                       u16(core::BsdfType::Lambertian), core::NO_MEDIUM, core::NO_LIGHT);

      const std::vector<vec3f> wv = {{-extent, floorY, backZ}, {extent, floorY, backZ},
                                     {extent, floorY + extent, backZ},
                                     {-extent, floorY + extent, backZ}};
      cfg.host.addMesh(wv.data(), 4, ff.data(), 2,
                       u16(core::BsdfType::Lambertian), core::NO_MEDIUM, core::NO_LIGHT);
    }

    // ——— Camera: lifted to p.camHeight and looking at the space between the
    // two objects' centres, so neither one is cut off and the floor does not
    // fill the frame. camHeight was accepted but unused before.
    const vec3f eye{0.0f, p.camHeight * R, 0.0f};
    const vec3f aim{0.0f, 0.5f * R, zObj};
    cfg.camera = core::makeCamera(eye, mps::normalize(aim - eye), {0.0f, 1.0f, 0.0f},
                                  p.fovDeg, p.width, p.height);

    mplog("buildCompareScene: dx={:.3f} radius={:.3f} res={}x{} | eye ({:.4f} {:.4f} {:.4f}) "
          "dir ({:.3f} {:.3f} {:.3f}) fov {:.1f} | left mesh bounds ({:.4f} {:.4f} {:.4f}).."
          "({:.4f} {:.4f} {:.4f})",
          dx, p.objectRadius, p.width, p.height,
          cfg.camera.pos.x, cfg.camera.pos.y, cfg.camera.pos.z,
          cfg.camera.dir.x, cfg.camera.dir.y, cfg.camera.dir.z, p.fovDeg,
          cfg.host.meshes[0].info.bounds.min.x, cfg.host.meshes[0].info.bounds.min.y,
          cfg.host.meshes[0].info.bounds.min.z,
          cfg.host.meshes[0].info.bounds.max.x, cfg.host.meshes[0].info.bounds.max.y,
          cfg.host.meshes[0].info.bounds.max.z);
    return cfg;
  }

} // namespace mps::host::m2g
