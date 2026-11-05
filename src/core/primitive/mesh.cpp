#include "core/primitive/primitive.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdparty/tiny_obj_loader.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mps::core {

  namespace {

    // Obj 里 position / normal / texcoord 的索引是分开的, 但 Mesh 只有一套
    // index, 所以按三元组去重成唯一顶点。
    struct VertexKey {
      int vertex;
      int normal;
      int texcoord;

      bool operator==(const VertexKey& other) const {
        return vertex == other.vertex && normal == other.normal &&
               texcoord == other.texcoord;
      }
    };

    struct VertexKeyHash {
      std::size_t operator()(const VertexKey& key) const noexcept {
        std::size_t h1 = std::hash<int>{}(key.vertex);
        std::size_t h2 = std::hash<int>{}(key.normal);
        std::size_t h3 = std::hash<int>{}(key.texcoord);
        std::size_t h = h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        return h ^ (h3 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
      }
    };

    vec3f face_normal(const vec3f& a, const vec3f& b, const vec3f& c) {
      vec3f ab{b.x - a.x, b.y - a.y, b.z - a.z};
      vec3f ac{c.x - a.x, c.y - a.y, c.z - a.z};
      vec3f n{
        ab.y * ac.z - ab.z * ac.y,
        ab.z * ac.x - ab.x * ac.z,
        ab.x * ac.y - ab.y * ac.x,
      };
      f32 len = n.x * n.x + n.y * n.y + n.z * n.z;
      if (len <= 0.0f) return vec3f{0.0f, 0.0f, 0.0f};
      f32 inv = 1.0f / std::sqrt(len);
      return vec3f{n.x * inv, n.y * inv, n.z * inv};
    }

  } // namespace

  MPH MeshHost buildMeshHost(std::string path) {
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;   // 保证每个面都是三角形
    config.vertex_color = false; // 不需要顶点色
    // 纹理/mtl 一律忽略: 不读取 reader.GetMaterials(), 不加载贴图。
    // 只保留 uv 坐标本身, 供之后自己贴图/采样使用。

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path, config)) {
      std::string msg = "Failed to load obj: " + path;
      if (!reader.Error().empty()) msg += "\n" + reader.Error();
      throw std::runtime_error(msg);
    }

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    const std::size_t num_positions = attrib.vertices.size() / 3;
    const std::size_t num_normals = attrib.normals.size() / 3;
    const std::size_t num_texcoords = attrib.texcoords.size() / 2;

    if (num_positions > UINT32_MAX) {
      throw std::runtime_error("Obj has too many vertices: " + path);
    }

    MeshHost ret{};

    // 记录每个唯一顶点是否自带法线; 没有的话后面用面法线做平滑法线。
    std::vector<u8> has_normal;

    std::unordered_map<VertexKey, u32, VertexKeyHash> remap;
    remap.reserve(num_positions);

    for (const auto& shape : shapes) {
      const auto& mesh = shape.mesh;
      std::size_t offset = 0; // 当前面在 mesh.indices 中的起点

      for (u32 nfv : mesh.num_face_vertices) {
        if (nfv != 3) {
          throw std::runtime_error(
              "Obj face is not triangulated (got " + std::to_string(nfv) +
              " vertices): " + path);
        }

        u32 tri[3];
        for (u32 k = 0; k < 3; ++k) {
          const tinyobj::index_t& idx = mesh.indices[offset + k];
          if (idx.vertex_index < 0 ||
              static_cast<std::size_t>(idx.vertex_index) >= num_positions) {
            throw std::runtime_error("Obj has invalid vertex index: " + path);
          }

          const bool use_normal =
              idx.normal_index >= 0 &&
              static_cast<std::size_t>(idx.normal_index) < num_normals;
          const bool use_uv =
              idx.texcoord_index >= 0 &&
              static_cast<std::size_t>(idx.texcoord_index) < num_texcoords;

          const VertexKey key{
              idx.vertex_index,
              use_normal ? idx.normal_index : -1,
              use_uv ? idx.texcoord_index : -1,
          };

          auto it = remap.find(key);
          if (it == remap.end()) {
            if (ret.vertices.size() >= UINT32_MAX) {
              throw std::runtime_error("Mesh has too many vertices: " + path);
            }

            const std::size_t vi =
                static_cast<std::size_t>(idx.vertex_index) * 3;
            ret.vertices.push_back(vec3f{
                attrib.vertices[vi + 0],
                attrib.vertices[vi + 1],
                attrib.vertices[vi + 2],
            });

            if (use_normal) {
              const std::size_t n =
                  static_cast<std::size_t>(idx.normal_index) * 3;
              ret.normals.push_back(vec3f{
                  attrib.normals[n + 0],
                  attrib.normals[n + 1],
                  attrib.normals[n + 2],
              });
            } else {
              ret.normals.push_back(vec3f{0.0f, 0.0f, 0.0f});
            }

            if (use_uv) {
              const std::size_t t =
                  static_cast<std::size_t>(idx.texcoord_index) * 2;
              ret.uvs.push_back(vec2f{
                  attrib.texcoords[t + 0],
                  attrib.texcoords[t + 1],
              });
            } else {
              ret.uvs.push_back(vec2f{0.0f, 0.0f});
            }

            has_normal.push_back(use_normal ? u8{1} : u8{0});

            it = remap
                     .emplace(key, static_cast<u32>(ret.vertices.size() - 1))
                     .first;
          }

          tri[k] = it->second;
        }

        ret.indices.push_back(tri[0]);
        ret.indices.push_back(tri[1]);
        ret.indices.push_back(tri[2]);
        offset += 3;

        // 对没有自带法线的顶点累加面法线(面积加权, 即未归一化的叉积)。
        const vec3f fn = face_normal(
            ret.vertices[tri[0]], ret.vertices[tri[1]], ret.vertices[tri[2]]);
        for (u32 k = 0; k < 3; ++k) {
          if (!has_normal[tri[k]]) {
            ret.normals[tri[k]].x += fn.x;
            ret.normals[tri[k]].y += fn.y;
            ret.normals[tri[k]].z += fn.z;
          }
        }
      }
    }

    // 归一化所有法线; 退化三角形的顶点回退到 +Y。
    for (vec3f& n : ret.normals) {
      f32 len2 = n.x * n.x + n.y * n.y + n.z * n.z;
      if (len2 <= 0.0f) {
        n = vec3f{0.0f, 1.0f, 0.0f};
      } else {
        f32 inv = 1.0f / std::sqrt(len2);
        n = vec3f{n.x * inv, n.y * inv, n.z * inv};
      }
    }

    ret.info.type = PrimitiveType::Mesh;
    ret.info.bsdfID = 0; // 不涉及材质/纹理, 由调用方后续指定
    ret.info.hasEmis = 0;
    ret.info.emisTexID = 0;
    ret.info.wpos = vec3f{0.0f, 0.0f, 0.0f};
    ret.info.quater = vec4f{0.0f, 0.0f, 0.0f, 1.0f}; // 单位四元数
    ret.info.scale = 1.0f;

    return ret;
  }

} // namespace mps::core
