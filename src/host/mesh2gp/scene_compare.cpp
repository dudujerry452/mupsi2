#include "host/mesh2gp/scene_compare.h"

#include <vector>

#include "core/bsdf/bsdf.h"
#include "core/utils/log.h"
#include "host/mesh2gp/gpisfile.h"
#include "host/obj_loader.h"

namespace mps::host::m2g {

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

    // ——— Layout ———
    // The artifact is the authority for scale: it was baked at a fixed size and
    // its voxel lattice CANNOT be rescaled at load time without desynchronising
    // the field from the geometry it samples. So the LEFT mesh is scaled by the
    // very same factor the artifact was baked with (header.meshScale) — then
    // both objects are literally the same geometry at the same size, which is
    // the whole point of the comparison.
    const f32 floorY = -150.0f;   // matches scene_builder

    const core::AABB artBox = art.bounds();
    const vec3f artCentre   = artBox.centroid();
    const f32   artRadius   = 0.5f * mps::norm(artBox.diagonal());
    // Objects are drawn at the artifact's own size; --object-radius can override
    // only if it agrees (the lattice cannot be rescaled).
    const f32 R = (p.objectRadius > 0.0f) ? p.objectRadius : artRadius;
    if (mps::fabsf(R - artRadius) > 1e-3f * mps::maxf(1.0f, artRadius)) {
      mplog("buildCompareScene: ERROR objectRadius {:.4f} != artifact radius {:.4f}. "
            "Re-bake with --scale so the lattice stays 1:1 with the geometry.", R, artRadius);
      cfg.hasMedium = false;
      return cfg;
    }

    const f32 dx   = 0.5f * p.gap * R;
    const f32 zObj = -(p.camDist * R);          // objects sit in front of the camera
    const vec3f leftTarget {-dx, 0.0f, zObj};
    const vec3f rightTarget{ dx, 0.0f, zObj};

    // ——— LEFT: plain triangle mesh (hard surface, no medium) ———
    {
      // Scale by the artifact's baked factor, then centre+drop exactly like the
      // right-hand object, so the two are the same size and share a floor.
      std::vector<vec3f> v = obj.vertices;
      const f32 bakedScale = art.header.meshScale > 0.0f ? art.header.meshScale : 1.0f;
      for (vec3f& x : v) x = x * bakedScale;

      core::AABB mb = core::AABB::empty();
      for (const vec3f& x : v) mb.extend(x);
      const vec3f mc = mb.centroid();
      const vec3f shift{ leftTarget.x - mc.x, floorY - mb.min.y, leftTarget.z - mc.z };
      for (vec3f& x : v) x = x + shift;

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
      // Move the artifact onto the right-hand slot and rest it on the SAME floor
      // plane as the left mesh.
      const vec3f offset{ rightTarget.x - artCentre.x,
                          floorY - artBox.min.y,
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
      // Must be the SAME plane the objects rest on (outer floorY). This used to
      // shadow it with -300, so the objects stood 150 units above the floor.
      const f32 backZ = -850.0f, extent = 500.0f;
      const f32 zNear = zObj + 300.0f;
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

    // ——— Camera: frame the two objects' real bounds.
    //
    // Aim at their combined centre, then pick a distance that actually FITS
    // them vertically. The old code used a fixed camDist and a height guessed
    // from R; both objects ended up hanging off the bottom of the frame because
    // the vertical half-extent at that distance (z*tan(fov/2)) was smaller than
    // the distance from the centre down to the floor.
    core::AABB both = core::AABB::empty();
    for (const core::Mesh& m : cfg.host.meshes) {
      if (m.info.bounds.max.y - m.info.bounds.min.y > 1.0f)   // skip floor/wall
        both = core::AABB{ {mps::minf(both.min.x, m.info.bounds.min.x),
                            mps::minf(both.min.y, m.info.bounds.min.y),
                            mps::minf(both.min.z, m.info.bounds.min.z)},
                           {mps::maxf(both.max.x, m.info.bounds.max.x),
                            mps::maxf(both.max.y, m.info.bounds.max.y),
                            mps::maxf(both.max.z, m.info.bounds.max.z)} };
    }
    const vec3f aim = both.centroid();
    const vec3f eye{0.0f, aim.y + p.camHeight * R, 0.0f};

    // Required half-extents, with a margin so neither object touches the border.
    const f32 halfH = 0.5f * (both.max.y - both.min.y);
    const f32 halfW = 0.5f * (both.max.x - both.min.x);
    const f32 tanV  = mps::tanf(0.5f * p.fovDeg * mps::pi / 180.0f);
    const f32 aspect = f32(p.width) / f32(p.height);
    const f32 needV = (halfH + mps::fabsf(aim.y - eye.y)) * p.frameMargin;
    const f32 needH = halfW * p.frameMargin;
    const f32 minDist = mps::maxf(needV / tanV, needH / (tanV * aspect));

    // camDist is a lower bound / override: never place the camera closer than
    // what is needed to fit both objects.
    const f32 dist = mps::maxf(mps::fabsf(aim.z - eye.z), minDist);
    const vec3f eyeFit{eye.x, eye.y, aim.z + dist};

    cfg.camera = core::makeCamera(eyeFit, mps::normalize(aim - eyeFit), {0.0f, 1.0f, 0.0f},
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
