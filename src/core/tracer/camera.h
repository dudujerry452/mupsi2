#pragma once

#include "core/geometry/ray.h"
#include "core/math/math_util.h"

namespace mps::core {

  // Orthonormal pinhole camera. Basis is precomputed on the host.
  struct Camera {
    vec3f pos;
    vec3f right, trueUp, dir;  // dir = view direction
    f32 tanHalfFov;
    f32 aspect;
    i32 width, height;

    MPHD Ray generateRay(i32 x, i32 y) const {
      f32 cx = (f32(x) + 0.5f) / f32(width);
      f32 cy = (f32(y) + 0.5f) / f32(height);
      f32 ndcX = (2.0f * cx - 1.0f) * aspect * tanHalfFov;
      f32 ndcY = (2.0f * cy - 1.0f) * tanHalfFov;

      Ray r;
      r.o = pos;
      r.dir = mps::normalize(ndcX * right + ndcY * trueUp + dir);
      r.min = 0.0f;
      r.max = 1e30f;
      return r;
    }
  };

  MPHD MPINL Camera makeCamera(const vec3f& pos, const vec3f& dir, const vec3f& up,
                               f32 fovDeg, i32 w, i32 h) {
    Camera c;
    c.pos = pos;
    c.dir = mps::normalize(dir);
    c.right = mps::normalize(mps::cross(c.dir, up));
    c.trueUp = mps::normalize(mps::cross(c.right, c.dir));
    c.tanHalfFov = mps::tanf(0.5f * fovDeg * mps::pi / 180.0f);
    c.aspect = f32(w) / f32(h);
    c.width = w;
    c.height = h;
    return c;
  }

}
