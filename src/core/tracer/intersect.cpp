#include "core/tracer/intersect.h"

#include "core/math/math_util.h"

namespace mps::core {

  MPHD HitRecord Intersect<PrimitiveType::Sphere>::run(const Ray& ray, const Sphere& s) {
    HitRecord hr;
    hr.hit = false;

    const vec3f C = s.info.wpos;
    const vec3f O = ray.o;
    const vec3f D = ray.dir;       // normalized
    const f32   R = s.info.scale;  // radius

    vec3f V = O - C;
    f32 k = mps::dot(D, V);
    f32 disc = k * k - (mps::dot(V, V) - R * R);
    if (disc < 0.0f) return hr;

    disc = mps::sqrtf(disc);
    f32 t1 = -k - disc;
    f32 t2 = -k + disc;

    f32 t;
    if (t1 >= ray.min && t1 <= ray.max)      t = t1;
    else if (t2 >= ray.min && t2 <= ray.max) t = t2;
    else return hr;

    vec3f p = ray.at(t);
    vec3f ng = mps::normalize(p - C);
    if (mps::dot(ng, D) > 0.0f) ng = -ng;  // face the ray

    // ——— spherical uv + tangent frame (same convention as original mupsi) ———
    vec3f local_ng = mps::quat_rotate_inv(s.info.quater, ng);
    vec3f T = (mps::fabsf(local_ng.y) > 0.999f)
                ? vec3f{1.0f, 0.0f, 0.0f}
                : vec3f{-local_ng.z, 0.0f, local_ng.x};
    T = mps::normalize(T);
    vec3f B = mps::cross(local_ng, T);
    B = (mps::dot(B, vec3f{0.0f, -1.0f, 0.0f}) < 0.0f) ? -B : B;

    vec3f tan   = mps::quat_rotate(s.info.quater, T);
    vec3f bitan = mps::quat_rotate(s.info.quater, B);

    f32 u = (mps::atan2f(local_ng.z, local_ng.x) + mps::pi) * mps::inv_two_pi;
    f32 v = mps::acosf(local_ng.y) * mps::inv_pi;

    hr.hit = true;
    hr.t = t;
    hr.p = p;
    hr.wo = -D;
    hr.ng = ng;
    hr.tan = tan;
    hr.bitan = bitan;
    hr.uv = {u, v};
    hr.texId = s.info.texId;
    hr.bsdfId = s.info.bsdfId;
    hr.mediumId = s.info.mediumId;
    hr.lightId = s.info.lightId;
    return hr;
  }

  MPHD HitRecord Intersect<PrimitiveType::Mesh>::run(const Ray& ray, const Mesh& m, const MeshGeometry& g) {
    HitRecord hr;
    hr.hit = false;

    const BVHNode* nodes = g.bvhNodes + m.bvhNodeOffset;
    const u32*     tris  = g.bvhTriIndices + m.bvhTriOffset;

    u32 tri; f32 u, v;
    f32 t = intersectBVH(ray, nodes, tris, g.vertices, g.faces, m.bvhRoot, tri, u, v);
    if (t < 0.0f) return hr;

    u32 i0 = g.faces[3 * tri + 0];
    u32 i1 = g.faces[3 * tri + 1];
    u32 i2 = g.faces[3 * tri + 2];
    vec3f v0 = g.vertices[i0];
    vec3f v1 = g.vertices[i1];
    vec3f v2 = g.vertices[i2];

    vec3f ng = mps::normalize(mps::cross(v1 - v0, v2 - v0));
    if (mps::dot(ng, ray.dir) > 0.0f) ng = -ng;  // face the ray

    hr.hit = true;
    hr.t = t;
    hr.p = ray.at(t);
    hr.wo = -ray.dir;
    hr.ng = ng;
    hr.uv = {u, v};   // barycentric (planar uv until per-vertex texcoords land)
    hr.texId = m.info.texId;
    hr.bsdfId = m.info.bsdfId;
    hr.mediumId = m.info.mediumId;
    hr.lightId = m.info.lightId;
    return hr;
  }

}
