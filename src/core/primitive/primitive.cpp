#include "core/primitive/primitive.h"

#include "core/scene/scene.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace mps::core {

  MPHD Sphere buildSphere(vec3f pos, f32 r, u32 bsdfID) {
    Sphere ret{};
    ret.info.type = PrimitiveType::Sphere;
    ret.info.wpos = pos;
    ret.info.quater = vec4f{0, 0, 0, 1};
    ret.info.scale = 1.0f;
    ret.info.bsdfID = bsdfID;
    ret.radius = r;
    return ret;
  }

  MPH void MeshHost::transform(SceneDevice& scene) const {
    if (vertices.size() > UINT32_MAX || indices.size() > UINT32_MAX) {
      throw std::runtime_error("Mesh is too large for u32 counts");
    }
    if (!normals.empty() && normals.size() != vertices.size()) {
      throw std::runtime_error("Mesh normals size does not match vertices");
    }
    if (!uvs.empty() && uvs.size() != vertices.size()) {
      throw std::runtime_error("Mesh uvs size does not match vertices");
    }

    const u32 num_new_vertices = static_cast<u32>(vertices.size());
    const u32 num_new_indices = static_cast<u32>(indices.size());

    const u64 new_num_vertices =
        static_cast<u64>(scene.num_vertices) + num_new_vertices;
    const u64 new_num_indices =
        static_cast<u64>(scene.num_indices) + num_new_indices;
    const u64 new_num_meshes =
        static_cast<u64>(scene.num_meshes) + 1;

    if (new_num_vertices > UINT32_MAX) {
      throw std::runtime_error("Scene has too many vertices");
    }
    if (new_num_indices > UINT32_MAX) {
      throw std::runtime_error("Scene has too many indices");
    }
    if (new_num_meshes > UINT32_MAX) {
      throw std::runtime_error("Scene has too many meshes");
    }

    const u32 old_num_vertices = scene.num_vertices;
    const u32 old_num_indices = scene.num_indices;

    const u32 total_vertices = static_cast<u32>(new_num_vertices);
    const u32 total_indices = static_cast<u32>(new_num_indices);

    // 先全部分配成功, 再统一替换, 避免中途 throw 破坏 scene 的旧数据。
    std::unique_ptr<vec3f[]> new_vertices(new vec3f[total_vertices]);
    std::unique_ptr<vec3f[]> new_normals(new vec3f[total_vertices]);
    std::unique_ptr<vec2f[]> new_uvs(new vec2f[total_vertices]);
    std::unique_ptr<u32[]> new_indices(new u32[total_indices]);
    std::unique_ptr<MeshDevice[]> new_meshes(
        new MeshDevice[static_cast<u32>(new_num_meshes)]);

    // 拷贝 scene 里已经有的数据。
    if (old_num_vertices != 0) {
      std::copy(
          scene.vertices, scene.vertices + old_num_vertices,
          new_vertices.get());
      std::copy(
          scene.normals, scene.normals + old_num_vertices,
          new_normals.get());
      std::copy(
          scene.uvs, scene.uvs + old_num_vertices,
          new_uvs.get());
    }
    if (old_num_indices != 0) {
      std::copy(
          scene.indices, scene.indices + old_num_indices,
          new_indices.get());
    }
    if (scene.num_meshes != 0) {
      std::copy(
          scene.meshes, scene.meshes + scene.num_meshes,
          new_meshes.get());
    }

    // 追加本 mesh 的数据; normals / uvs 缺失时补零, 保证全局数组等长。
    std::copy(vertices.begin(), vertices.end(), new_vertices.get() + old_num_vertices);
    if (normals.empty()) {
      std::fill(
          new_normals.get() + old_num_vertices,
          new_normals.get() + total_vertices,
          vec3f{0.0f, 0.0f, 0.0f});
    } else {
      std::copy(
          normals.begin(), normals.end(),
          new_normals.get() + old_num_vertices);
    }
    if (uvs.empty()) {
      std::fill(
          new_uvs.get() + old_num_vertices,
          new_uvs.get() + total_vertices,
          vec2f{0.0f, 0.0f});
    } else {
      std::copy(
          uvs.begin(), uvs.end(),
          new_uvs.get() + old_num_vertices);
    }
    std::copy(
        indices.begin(), indices.end(),
        new_indices.get() + old_num_indices);

    MeshDevice mesh_dev{};
    mesh_dev.info = info;
    mesh_dev.num_vertices = num_new_vertices;
    mesh_dev.offset_vertices = old_num_vertices;
    mesh_dev.num_indices = num_new_indices;
    mesh_dev.offset_indices = old_num_indices;
    new_meshes[scene.num_meshes] = mesh_dev;

    // 全部成功, 提交。
    delete[] scene.vertices;
    delete[] scene.normals;
    delete[] scene.uvs;
    delete[] scene.indices;
    delete[] scene.meshes;

    scene.vertices = new_vertices.release();
    scene.normals = new_normals.release();
    scene.uvs = new_uvs.release();
    scene.indices = new_indices.release();
    scene.meshes = new_meshes.release();
    scene.num_vertices = total_vertices;
    scene.num_indices = total_indices;
    scene.num_meshes = static_cast<u32>(new_num_meshes);
  }

} // namespace mps::core
