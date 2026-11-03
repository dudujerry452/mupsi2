#include "core/scene/scene.h"
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
    } 

    if(num_bsdfs > 0) {
      auto dev = ::mps::upload(bsdfs, num_bsdfs); 
      bsdfs = dev; 
    }
  }
  
  MPH void SceneDevice::destory() {
    if(num_spheres != 0) {
      ::mps::destory(spheres); 
    }
    
    if(num_textures != 0) {
      for(u32 i = 0; i < num_textures; i ++) {
        ::mps::destory(textures[i].pixels); 
      }
      ::mps::destory(textures); 
    }

    if(num_bsdfs != 0) {
      ::mps::destory(bsdfs); 
    }

    *this = SceneDevice{};
  }


  MPH void SceneDevice::free() {
    if(spheres != nullptr) delete[] spheres; 

    if(num_textures != 0) {
      for(u32 i = 0; i < num_textures; i ++) {
        delete [] textures[i].pixels; 
      }
      delete[] textures; 
    }

    if(num_bsdfs != 0) {
      delete[] bsdfs; 
    }

    *this = SceneDevice{}; 
  }

  MPH SceneDevice SceneHost::transform() const {
    core::SceneDevice dev{};

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
        memcpy(dev.textures[i].pixels, textures[i].pixels, dev.textures[i].bytesize()); 
      }
    }

    dev.num_bsdfs = bsdfs.size(); 
    if(dev.num_bsdfs != 0) {
      dev.bsdfs = new core::Bsdf[dev.num_bsdfs](); 
      std::copy(bsdfs.begin(), bsdfs.end(), dev.bsdfs);
    }

    return dev; 
  }

  MPH u32 SceneHost::AddSphere(Sphere&& sphere) {
    spheres.push_back(std::move(sphere)); 
    return spheres.size()-1; 
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
    return textures.size()-1; 
  }
  MPH u32 SceneHost::AddBsdf(BsdfType type, u32 texID) {
    bsdfs.emplace_back(type, texID); 
    return bsdfs.size()-1; 
  }

  MPH SceneHost buildScene_Spot() {
    SceneHost scene; 
    
    u32 texid = scene.AddTexture("spot_tex1.png");
    u32 bsdfid = scene.AddBsdf(BsdfType::Lambertain, texid); 
    
    scene.AddSphere(buildSphere({0, 0, -10}, 2, bsdfid));
    return scene; 
  }

}