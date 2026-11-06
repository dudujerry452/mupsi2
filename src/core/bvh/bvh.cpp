#include "core/bvh/bvh.h"
#include <cmath>
#include "core/basic/types.h"
#include "core/tracer/constant.h"

namespace mps::core {

  using TRIS_CEN = std::vector<std::pair<u32, vec3f>>; 

  struct build_bvh_context {
    std::vector<BVHNode> nodes; 
    std::vector<vec3f> triangles_centroid; 
    std::vector<u32> triangles_order; 
  }; 

  MPH void SceneDevice::buildBVH() {
    bvh_b.instances_order = new u32[num_triangles];
    for(u32 i = 0; i < num_triangles; i ++) {
      bvh_b.instances_order[i] = i; 
    } 
  }

  MPH void compute_triangles_centroid(build_bvh_context& ctx, SceneDevice &scene) {
    ctx.triangles_centroid.reserve(scene.num_triangles); 
    constexpr f32 inv_3 = 1.0f / 3.0f; 
    for(u32 i = 0; i < scene.num_triangles; i ++) {
      // centroid
      ctx.triangles_centroid.emplace_back((scene.vertices[scene.triangles[i].a]+ 
                      scene.vertices[scene.triangles[i].b]+ 
                    scene.vertices[scene.triangles[i].c]) * inv_3); 
    }
  }

  // triangles's centroid's AABB
  // start, count in triangles_order
  MPH AABB compute_triangles_AABB(u32 start, u32 count, const build_bvh_context& ctx) {
    vec3f first = ctx.triangles_centroid[ctx.triangles_order[start]]; 
    AABB ret = {first, first};  
    for(u32 i = start; i < start+count; i ++) {
      ret.update(ctx.triangles_centroid[ctx.triangles_order[i]]); 
    }
    return ret; 
  }


  // start, count in triangle_order, range from mesh
  MPH void buildBLAS_(u32 start, u32 count, build_bvh_context& ctx, SceneDevice &scene) {
    BVHNode& node = ctx.nodes[ctx.nodes.size()-1]; 
    node.aabb = compute_triangles_AABB(start, count, ctx); 
    node.first = start; 
    node.count = count; 
    if(count <= RenderConstants::MIN_BVH_NODE_COUNT) {
      return; 
    }
    // TODO: use median value to divide triangles
    BVHNode left_node{}; 
  }

  MPH void buildBLAS(u32 meshID, SceneDevice& scene) {
    


  }
}