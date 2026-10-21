#pragma once

#include "core/geometry/ray.h"
#include "core/math/math_util.h"

namespace mps::core {

  struct AABB {
    vec3f min, max;

    MPHD AABB() : min{ INFINITY,  INFINITY,  INFINITY },
                  max{-INFINITY, -INFINITY, -INFINITY } {}
    MPHD AABB(const vec3f& mn, const vec3f& mx) : min(mn), max(mx) {}

    MPHD static AABB empty() { return AABB(); }

    MPHD void extend(const vec3f& p) {
      min = mps::minv(min, p);
      max = mps::maxv(max, p);
    }

    MPHD vec3f centroid() const { return 0.5f * (min + max); }
    MPHD vec3f diagonal() const { return max - min; }

    MPHD f32 surfaceArea() const {
      vec3f d = diagonal();
      return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }

    MPHD i32 longestAxis() const {
      vec3f d = diagonal();
      if (d.x > d.y && d.x > d.z) return 0;
      if (d.y > d.z) return 1;
      return 2;
    }

    MPHD bool isValid() const {
      return min.x <= max.x && min.y <= max.y && min.z <= max.z;
    }

    MPHD AABB unite(const AABB& o) const {
      return {mps::minv(min, o.min), mps::maxv(max, o.max)};
    }

    // Slab method, bounded by the ray's [min, max] interval.
    MPHD bool intersect(const Ray& ray) const {
      f32 tmin = ray.min, tmax = ray.max;

      if (mps::fabsf(ray.dir.x) > 1e-8f) {
        f32 inv = 1.0f / ray.dir.x;
        f32 t1 = (min.x - ray.o.x) * inv;
        f32 t2 = (max.x - ray.o.x) * inv;
        if (t1 > t2) { f32 tmp = t1; t1 = t2; t2 = tmp; }
        tmin = mps::maxf(tmin, t1);
        tmax = mps::minf(tmax, t2);
        if (tmin > tmax) return false;
      } else if (ray.o.x < min.x || ray.o.x > max.x) {
        return false;
      }

      if (mps::fabsf(ray.dir.y) > 1e-8f) {
        f32 inv = 1.0f / ray.dir.y;
        f32 t1 = (min.y - ray.o.y) * inv;
        f32 t2 = (max.y - ray.o.y) * inv;
        if (t1 > t2) { f32 tmp = t1; t1 = t2; t2 = tmp; }
        tmin = mps::maxf(tmin, t1);
        tmax = mps::minf(tmax, t2);
        if (tmin > tmax) return false;
      } else if (ray.o.y < min.y || ray.o.y > max.y) {
        return false;
      }

      if (mps::fabsf(ray.dir.z) > 1e-8f) {
        f32 inv = 1.0f / ray.dir.z;
        f32 t1 = (min.z - ray.o.z) * inv;
        f32 t2 = (max.z - ray.o.z) * inv;
        if (t1 > t2) { f32 tmp = t1; t1 = t2; t2 = tmp; }
        tmin = mps::maxf(tmin, t1);
        tmax = mps::minf(tmax, t2);
        if (tmin > tmax) return false;
      } else if (ray.o.z < min.z || ray.o.z > max.z) {
        return false;
      }

      return true;
    }
  };

}
