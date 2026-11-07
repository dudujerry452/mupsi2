#include "core/bvh/bvh.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include "core/basic/types.h"
#include "core/tracer/constant.h"

namespace mps::core {

  using TRIS_CEN = std::vector<std::pair<u32, vec3f>>; 

  struct build_bvh_context_b {
    std::vector<BVHNode> nodes; 
    std::vector<vec3f> triangles_centroid; 
    std::vector<u32> triangles_order; 
    std::vector<u32> node_begin;
  }; 

  MPH void compute_triangles_centroid(build_bvh_context_b& ctx, SceneDevice &scene) {
    ctx.triangles_centroid.reserve(scene.num_triangles); 
    constexpr f32 inv_3 = 1.0f / 3.0f; 
    for(u32 i = 0; i < scene.num_triangles; i ++) {
      // centroid
      ctx.triangles_centroid.emplace_back((scene.vertices[scene.triangles[i].a]+ 
                      scene.vertices[scene.triangles[i].b]+ 
                    scene.vertices[scene.triangles[i].c]) * inv_3); 
    }
  }

  MPH void SceneDevice::buildBVH() {
    
    // initialize BLAS build
    build_bvh_context_b ctx; 
    ctx.triangles_centroid.reserve(num_triangles); 
    ctx.triangles_order.reserve(num_triangles); 
    ctx.nodes.push_back({}); 
    ctx.node_begin.resize(num_meshes); 
    for(u32 i = 0; i < num_triangles; i ++) {
      ctx.triangles_order[i] = i; 
    } 
    compute_triangles_centroid(ctx, *this); 

    // compute BLAS 
    for(u32 i = 0; i < num_meshes; i ++) {
      buildBLAS(i, *this); 
    }


    // copy data back to scene
    bvh_b.triangles_order = new u32[num_triangles];
    for(u32 i = 0; i < num_triangles; i ++) {
      bvh_b.triangles_order[i] = ctx.triangles_order[i];
    }
    bvh_b.bvh_node_begin = new u32[num_meshes]; 
    for(u32 i = 0; i < num_meshes; i ++) {
      bvh_b.bvh_node_begin[i] = ctx.node_begin[i]; 
    }
    bvh_b.num_bvhnode = ctx.nodes.size(); 
    bvh_b.bvh_nodes = new BVHNode[bvh_b.num_bvhnode]; 
    for(u32 i = 0; i < bvh_b.num_bvhnode; i ++) {
      bvh_b.bvh_nodes[i] = ctx.nodes[i]; 
    }
  }

  // triangles's centroid's AABB
  // start, count in triangles_order
  MPH AABB compute_triangles_AABB(u32 start, u32 count, const build_bvh_context_b& ctx) {
    vec3f first = ctx.triangles_centroid[ctx.triangles_order[start]]; 
    AABB ret = {first, first};  
    for(u32 i = start; i < start+count; i ++) {
      ret.update(ctx.triangles_centroid[ctx.triangles_order[i]]); 
    }
    return ret; 
  }



  MPH u32 select_longest_axis(u32 start, u32 count, AABB aabb, build_bvh_context_b& ctx) {
    byte flag = 0;
    vec3f len = aabb.length();  
    if(len.x >= len.y && len.x >= len.z); 
    else if(len.y >= len.x && len.y >= len.z) flag = 1; 
    else if(len.z >= len.x && len.z >= len.y) flag = 2; 
    const std::function<bool(u32, u32)> f[3] = {
      [&](u32 a, u32 b) {
        return ctx.triangles_centroid[a].x < ctx.triangles_centroid[b].x; 
      }, 
      [&](u32 a, u32 b) {
        return ctx.triangles_centroid[a].y < ctx.triangles_centroid[b].y; 
      }, 
      [&](u32 a, u32 b) {
        return ctx.triangles_centroid[a].z < ctx.triangles_centroid[b].z; 
      }   
    };  // TODO: test inline style's performance; 用汇编看看
    i32 mid = start + count/2; 
    std::nth_element(ctx.triangles_order.begin()+start,
        ctx.triangles_order.begin() + mid, 
        ctx.triangles_order.begin()+start+count,  
        f[flag]
      );
    return mid; 
  }

  // start, count in triangle_order, range from mesh
  MPH void buildBLAS_(u32 start, u32 count, build_bvh_context_b& ctx) {
    BVHNode& node = ctx.nodes.back(); 
    node.aabb = compute_triangles_AABB(start, count, ctx); 
    node.first = start; 
    node.count = count; 
    if(count <= RenderConstants::MIN_BVH_NODE_COUNT) {
      return; 
    }
    u32 mid = select_longest_axis(start, count, node.aabb, ctx); 
    ctx.nodes.push_back({});  /// BVHNode& node 失效
    node.left = ctx.nodes.size()-1; 
    buildBLAS_(start, mid-start, ctx);
    ctx.nodes.push_back({}); 
    node.right = ctx.nodes.size()-1;
    buildBLAS_(mid, count - (mid-start), ctx); 
  }

  MPH void buildBLAS(u32 meshID, build_bvh_context_b& ctx, SceneDevice& scene) {
    u32 start = scene.meshes[meshID].offset_triangles; 
    u32 count = scene.meshes[meshID].num_triangles; 
    
    buildBLAS_(start, count, ctx); 
  }
}