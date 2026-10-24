#pragma once

#include "core/basic/types.h"
#include "core/math/math_util.h"

namespace mps::core {

  MPHD void combine(u32& seed, u32 val);

  // usage: Random random(make_seed(<args>...))
  template<typename... Args>
  MPHD u32 make_seed(Args... args) {
    u32 h = 0;
    ((combine(h, static_cast<u32>(args)), ...));
    return h;
  }

  // faster make_seed 
  MPHD u32 make_seed_f(u32 x);
  MPHD u32 make_seed_f(u32 x, u32 y);
  MPHD u32 make_seed_f(u32 x, u32 y, u32 z);
  MPHD u32 make_seed_f(u32 x, u32 y, u32 z, u32 w);

  MPHD float u32_to_f(u32 x);

  // uint -> f32[0,1]
  MPHD MPINL f32 normalizedUint(u32 i) {
    union { u32 u; f32 f; } x;
    x.u = (i >> 9u) | 0x3F800000u;
    return x.f - 1.0f;
  }

  // PCG32 random
  struct Random {
    u64 state;
    u64 oldState;

    MPHD Random(u64 seed) : state(seed), oldState(seed) {
      if (state == 0) state = 1; // no warm up its good enough 
    }
    MPHD void setState(u64 s) { state = s; if (state == 0) state = 1; }

    MPHD u32 nextI() {
      oldState = state;
      state = oldState * 6364136223846793005ULL + 1;
      u32 xorShifted = u32(((oldState >> 18u) ^ oldState) >> 27u);
      u32 rot = u32(oldState >> 59u);
      return (xorShifted >> rot) | (xorShifted << (u32(-i32(rot)) & 31u));
    }
    MPHD u32 getI() const {
      u32 xorShifted = u32(((oldState >> 18u) ^ oldState) >> 27u);
      u32 rot = u32(oldState >> 59u);
      return (xorShifted >> rot) | (xorShifted << (u32(-i32(rot)) & 31u));
    }
    MPHD f32 next1D() { return normalizedUint(nextI()); }
    MPHD f32 get1D() const { return normalizedUint(getI()); }

    // Box-Muller: N(0,1)
    MPHD f32 standard_normal() {
      f32 u1 = 1.0f - next1D(); // (0, 1], safe for log
      f32 u2 = next1D();
      f32 r  = mps::sqrtf(-2.0f * mps::logf(u1));
      f32 a  = mps::two_pi * u2;
      return r * mps::cosf(a);
    }

    // ±1 with p=0.5
    MPHD f32 sign() { return next1D() < 0.5f ? -1.0f : 1.0f; }
  };

}
