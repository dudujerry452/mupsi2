#pragma once 

#include "core/basic/core.h"
#include "core/utils/panic.h"

#define ENUM_PRIMITIVE_FUNC(X) \
  X(Sphere) \
  X(Mesh)

namespace mps::core {

  #define DEF_ENUM(name) name, 

  enum class PrimitiveType: u32 {
    ENUM_PRIMITIVE_FUNC(DEF_ENUM)
  }; 

  #undef DEF_ENUM

  template<PrimitiveType T> 
  struct Primitive;

  template<>
  struct Primitive<PrimitiveType::Sphere> {
    PrimitiveType type; 

    f32 radius;

    u32 bsdfId;
    u32 texId;
  }; 


  template<> 
  struct Primitive<PrimitiveType::Mesh> {

    struct Triangle {
      u32 i[3]; 
    };

    PrimitiveType type; 

    u32 vertex_cnt; 
    u32 fac_cnt; 

    vec3f *vertices; 
    Triangle *faces; 

    u32 bsdfId; 
    u32 texId; 
  }; 


  template<PrimitiveType T>
  struct PrimitiveTrait {
    using Type = Primitive<T>; 
    static constexpr u32 raw_size = sizeof(Type);
    static constexpr u32 aligned_size = (raw_size+15)&(~15);  
  };

  inline u32 rawSizeOfPrimitive(PrimitiveType type) {
    #define DEF_TRAIT(name) \
    case PrimitiveType::name: {return PrimitiveTrait<PrimitiveType::name>::raw_size; }

    switch(type) {
      ENUM_PRIMITIVE_FUNC(DEF_TRAIT)
      default: { lpanic("Primitive not implemented");  } // panic
    }

    #undef DEF_TRAIT

  }




}