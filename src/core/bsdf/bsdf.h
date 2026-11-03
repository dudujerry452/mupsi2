#pragma once 
#include "core/basic/core.h"
#include "core/basic/types.h"
#include "core/utils/log.h"
#include "core/math/random.h"

namespace mps::core {

  // 单一数据源:每行一个类型,额外参数原样透传
  #define MPS_BSDF_TYPES(M, ...) \
    M(Lambertain, __VA_ARGS__)   \
    M(Specular,   __VA_ARGS__)

  // 枚举项
  #define MPS_ENUM_ITEM(n, ...) n,
  enum class BsdfType : u16 {
    MPS_BSDF_TYPES(MPS_ENUM_ITEM, )
  };
  #undef MPS_ENUM_ITEM

  struct Bsdf { BsdfType type; u32 texID; };
  // get it from trace()
  struct BsdfParam { BsdfType type; RGB_F32 albedo;  }; 

  template<BsdfType T> struct BsdfImpl;  // 主模板

  // 未特化类型的默认 panic 实现
  template<BsdfType T> struct BsdfImpl {
    static MPHD MPINL RGB_F32 f(vec3f, vec3f, const BsdfParam&) { mppanic("bsdf f"); return {}; }
    static MPHD MPINL f32     pdf(vec3f, vec3f, const BsdfParam&) { mppanic("bsdf pdf"); return 0; }
    static MPHD MPINL vec3f   sample(vec3f, f32&, const BsdfParam&, Random& rd) { mppanic("bsdf sample"); return {}; }
  };

  // ------------------ 

  template<> struct BsdfImpl<BsdfType::Lambertain> {
    static MPHD RGB_F32 f(vec3f wi, vec3f wo, const BsdfParam& par);
    static MPHD f32     pdf(vec3f wi, vec3f wo, const BsdfParam& par);
    static MPHD vec3f   sample(vec3f wo, f32& pdf, const BsdfParam& par, Random& rd);
  };

  // ------------------ 

  // 分派宏:CALL 通过 MPS_BSDF_TYPES 真正传进 MPS_BSDF_CASE
  #define MPS_BSDF_CASE(n, CALL) \
    case BsdfType::n: return BsdfImpl<BsdfType::n>::CALL;

  #define MPS_BSDF_DISPATCH(RET, CALL)                              \
    switch (par.type) {                                           \
      MPS_BSDF_TYPES(MPS_BSDF_CASE, CALL)                           \
      default: mppanic("bad BsdfType"); return RET{};               \
    }

  MPHD MPINL RGB_F32 bsdf_f(vec3f wi, vec3f wo, const BsdfParam& par) {
    MPS_BSDF_DISPATCH(RGB_F32, f(wi, wo, par))
  }
  MPHD MPINL f32 bsdf_pdf(vec3f wi, vec3f wo, const BsdfParam& par) {
    MPS_BSDF_DISPATCH(f32, pdf(wi, wo, par))
  }
  MPHD MPINL vec3f bsdf_sample(vec3f wo, f32& pdf, const BsdfParam& par, Random& rd) {
    MPS_BSDF_DISPATCH(vec3f, sample(wo, pdf, par, rd))
  }

  #undef MPS_BSDF_DISPATCH
  #undef MPS_BSDF_CASE
  #undef MPS_BSDF_TYPES

} // namespace mps