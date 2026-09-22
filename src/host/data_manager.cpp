#include "host/data_manager.h"

#include "core/primitive/primitive.h"
#include <vector> 
#include <stdexcept> 
#include <cstring> 
#include <memory>


namespace mps::host {

  std::tuple<u32, u32*, void*> flatten_primitive(const std::vector<void*>& prims) {
    if(prims.empty()) return {0, nullptr, nullptr}; 
    
    u32 num = prims.size(); 
    u32* starts = new u32[num];  starts[0] = 0; 
    std::vector<u32> lens(num, 0); 

    i32 i = 0, cnt = 0; 
    for(auto p: prims) {
      auto* ptr = reinterpret_cast<core::PrimitiveInfo*>(p);
      u32 siz = core::alignedSizeOfPrimitive(ptr->type); 
      starts[i] = cnt; 
      cnt += siz; 
      lens[i] = siz; 
    }

    char* flat = new char[cnt]; 
    for(u32 i = 0; i < num; i ++) {
      memcpy(flat+starts[i], prims[i], lens[i]); 
    }
    return {num, starts, static_cast<void*>(flat)};  
  }

}