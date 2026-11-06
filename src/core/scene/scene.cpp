#include "core/scene/scene.h"
#include "core/basic/mems.h"
#include "core/basic/types.h"
#include "core/bsdf/bsdf.h"
#include "core/primitive/primitive.h"

#include "core/utils/cuda_util.h"

#include <string> 
#include <cstring> 
#include <memory>

#include "src/host/img_loader.h"

namespace mps::core {

  
  MPH void SceneDevice::upload() {
    if(num_vertices > 0) {
      vertices = ::mps::upload(vertices, num_vertices);
      normals = ::mps::upload(normals, num_vertices);
      uvs = ::mps::upload(uvs, num_vertices);
    }

    if(num_triangles > 0) {
      triangles = ::mps::upload(triangles, num_triangles);
      bvh_b.instances_order = ::mps::upload(bvh_b.instances_order, num_triangles);
    }

    if(num_meshes > 0) {
      meshes = ::mps::upload(meshes, num_meshes);
    }

    if(num_spheres > 0) {
      auto dev = ::mps::upload(spheres, num_spheres);
      spheres = dev;  
    }
    
    if(num_textures > 0) {
      using textype = std::remove_pointer_t<decltype(textures)>; 
      textype *texs = new textype[num_textures]();  
      for(u32 i = 0; i < num_textures; i ++) {
        texs[i] = textures[i]; 
        texs[i].upload(); 
      }
      textures = ::mps::upload(texs, num_textures); 
      safe_free_arr(&texs);  
    } 

    if(num_bsdfs > 0) {
      auto dev = ::mps::upload(bsdfs, num_bsdfs); 
      bsdfs = dev; 
    }

    if(num_lights > 0) {
      auto dev = ::mps::upload(lights, num_lights); 
      lights = dev; 
    }
  }
  
  MPH void SceneDevice::destory() {
    if(num_vertices != 0) {
      ::mps::destory(vertices);
      ::mps::destory(normals);
      ::mps::destory(uvs);
    }

    if(num_triangles != 0) {
      ::mps::destory(triangles);
      ::mps::destory(bvh_b.instances_order); 
    }

    if(num_meshes != 0) {
      ::mps::destory(meshes);
    }

    if(num_spheres != 0) {
      ::mps::destory(spheres); 
    }
    
    if(num_textures != 0) {
      // download texs from device first
      auto* host_texs = ::mps::download(textures, num_textures);
      for(u32 i = 0; i < num_textures; i ++) {
        ::mps::destory(host_texs[i].pixels); 
      }
      safe_free_arr(&host_texs);  
      ::mps::destory(textures); 
    }

    if(num_bsdfs != 0) {
      ::mps::destory(bsdfs); 
    }

    if(num_lights != 0) {
      ::mps::destory(lights); 
    }

    *this = SceneDevice{};
  }


  MPH void SceneDevice::free() {
    if(num_vertices != 0) {
      safe_free_arr(&vertices);
      safe_free_arr(&normals);
      safe_free_arr(&uvs);
    }

    if(num_triangles != 0) {
      safe_free_arr(&triangles);
      safe_free_arr(&bvh_b.instances_order); 
    }

    if(num_meshes != 0) {
      safe_free_arr(&meshes);
    }

    if(num_spheres != 0) {
      safe_free_arr(&spheres); 
    }

    if(num_textures != 0) {
      for(u32 i = 0; i < num_textures; i ++) {
        safe_free_arr(&textures[i].pixels);  
      }
      safe_free_arr(&textures); 
    }

    if(num_bsdfs != 0) {
      safe_free_arr(&bsdfs); 
    }

    if(num_lights != 0) {
      safe_free_arr(&lights); 
    }

    *this = SceneDevice{}; 
  }

  MPH SceneDevice SceneHost::transform() const {
    core::SceneDevice dev{};

    // 每个 MeshHost 把数据追加到 dev 的全局顶点/索引 buffer,
    // MeshDevice 里只保存 offset + count。
    for(const auto& mesh : meshes) {
      mesh.transform(dev);
    }

    dev.num_spheres = spheres.size(); 
    if(dev.num_spheres != 0) {
      dev.spheres = new core::Sphere[dev.num_spheres]();
      std::copy(spheres.begin(), spheres.end(), dev.spheres);
    }

    dev.num_textures = textures.size(); 
    if(dev.num_textures != 0) {
      dev.textures = new Texture<RGB_F32>[dev.num_textures](); 
      for(u32 i = 0; i < dev.num_textures; i ++) {
        dev.textures[i] = textures[i]; 
        dev.textures[i].pixels = new RGB_F32[dev.textures->pixelnum()](); 
        memcpy(dev.textures[i].pixels, textures[i].pixels, dev.textures[i].bytesize()); 
      }
    }

    dev.num_bsdfs = bsdfs.size(); 
    if(dev.num_bsdfs != 0) {
      dev.bsdfs = new core::Bsdf[dev.num_bsdfs](); 
      std::copy(bsdfs.begin(), bsdfs.end(), dev.bsdfs);
    }

    dev.num_lights = lights.size(); 
    if(dev.num_lights != 0) {
      dev.lights = new core::Light[dev.num_lights](); 
      std::copy(lights.begin(), lights.end(), dev.lights); 
    }

    return dev; 
  }

  // might be called multi times, so safe realease
  MPH SceneHost::~SceneHost() {
    for(auto& tex: textures) {
      safe_free_arr(&tex.pixels);   
    }
  }

  MPH u32 SceneHost::AddSphere(Sphere&& sphere) {
    spheres.push_back(std::move(sphere)); 
    u32 id = spheres.size()-1; 
    if(sphere.info.hasEmis) {
      lights.push_back({PrimitiveType::Sphere, id}); 
    }
    return id; 
  }

  MPH u32 SceneHost::AddTexture(const char* path) {
    auto tex = host::load_albedo(path); 
    textures.push_back(tex); 
    return textures.size()-1; 
  }
  MPH u32 SceneHost::AddTexture(i32 width, i32 height) {
    auto tex = Texture<RGB_F32>{}; 
    tex.width = width; 
    tex.height = height; 
    tex.pixels = new RGB_F32[tex.pixelnum()](); 
    textures.push_back(tex); 
    return textures.size()-1; 
  }
  MPH u32 SceneHost::AddBsdf(BsdfType type, u32 texID) {
    bsdfs.emplace_back(type, texID); 
    return bsdfs.size()-1; 
  }

  MPH SceneHost buildScene_Spot() {
    SceneHost scene; 
    
    u32 texid = scene.AddTexture("spot_tex1.jpg");
    u32 bsdfid = scene.AddBsdf(BsdfType::Lambertain, texid); 
    
    scene.AddSphere(buildSphere({0, 0, -10}, 2, bsdfid));
    Sphere sph = buildSphere({-4, 0, -6}, 2, bsdfid);
    sph.info.hasEmis = 1; 
    sph.info.emisTexID = texid; 
    scene.AddSphere(std::move(sph));
    return scene; 
  }

}