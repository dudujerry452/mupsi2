#pragma once 

#include "core/basic/core.h"
#include "core/utils/panic.h"

#define ENUM_PRIMITIVE_FUNC(X) \
  X(Sphere) \
  X(Mesh)

namespace mps::core {

  // ----------- main template definations  ----------------

  #define DEF_ENUM(name) name, 

  enum class PrimitiveType: u16 {
    ENUM_PRIMITIVE_FUNC(DEF_ENUM)
  }; 

  #undef DEF_ENUM

  template<PrimitiveType T> 
  struct Primitive;

  struct alignas(16) PrimitiveInfo {
    PrimitiveType type; 
    u16 bsdfId;
    u16 texId;
    vec3f wpos;
    vec4f quater; 
  }; 

  // ---------- primitive definations ------------

  template<>
  struct Primitive<PrimitiveType::Sphere> {
    PrimitiveInfo info; 

    f32 radius;
  }; 


  template<> 
  struct Primitive<PrimitiveType::Mesh> {
    PrimitiveInfo info; 

    u32 vertex_cnt; 
    u32 fac_cnt; 

    vec3f *vertices; 
    struct Triangle {
      u32 i[3]; 
    }*faces;

  }; 


  // ---------- traits ---------

  template<PrimitiveType T>
  struct PrimitiveTrait {
    using Type = Primitive<T>; 
    static constexpr u32 raw_size = sizeof(Type);
    static constexpr u32 aligned_size = (raw_size+15)&(~15);  
  };

  inline u32 alignedSizeOfPrimitive(PrimitiveType type) {
    #define DEF_TRAIT(name) \
    case PrimitiveType::name: {return PrimitiveTrait<PrimitiveType::name>::aligned_size; }

    switch(type) {
      ENUM_PRIMITIVE_FUNC(DEF_TRAIT)
      default: { lpanic("Primitive not implemented");  } // panic
    }

    #undef DEF_TRAIT
  }

}