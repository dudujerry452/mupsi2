#pragma once 

#include "core/primitive/primitive.h"
#include <vector> 
#include <stdexcept> 
#include <cstring> 
#include <memory>

namespace mps::host {

  // return starts(heap) flat(bytes)
  std::pair<u32*, void*> flatten_primitive(const std::vector<void*>& prims) {
    struct PrimitiveTypeStruct {
    mps::core::PrimitiveType type; 
    };

    if(prims.empty()) return {nullptr, nullptr}; 
    
    u32 num = prims.size(); 
    u32* starts = new u32[num];  
    starts[0] = 0; 
    std::vector<u32> lens(num, 0); 
    i32 i = 0, cnt = 0; 
    for(auto p: prims) {
    auto* ptr = reinterpret_cast<PrimitiveTypeStruct*>(p); 
    auto type = ptr->type; 
    u32 psiz = 0; 
    switch(type) {
      case core::PrimitiveType::Sphere: {
        psiz = sizeof(core::Primitive<core::PrimitiveType::Sphere>); 
        break; 
      }
      case core::PrimitiveType::Mesh: {
        psiz = sizeof(core::Primitive<core::PrimitiveType::Mesh>);
        break; 
      }
      default: {
        throw std::runtime_error("flatten_primitive: invalid primitive struct type");
      }
    }
    cnt += psiz; 
    lens[i] = psiz; 
    if(i < num-1) starts[++i] = cnt;
    }
    char* flat = new char[cnt]; 
    for(i32 i = 0; i < num; i ++) {
      memcpy(flat+starts[i], prims[i], lens[i]); 
    }
    return {starts, static_cast<void*>(flat)};  
}

}