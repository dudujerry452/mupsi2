#pragma once 

#include "core/basic/core.h"
#include "core/math/math_util.h"


namespace mps::core {

struct Camera {
    vec3f wpos;
    vec4f quat; // (x, y, z, w)
    f32 fov;
    f32 znear; 
    f32 aspect;

    MPHD void LookAt(vec3f target, vec3f up) {
        vec3f forward = mps::normalize(target - wpos);
        vec3f right = mps::normalize(mps::cross(forward, up));
        up = mps::normalize(mps::cross(right, forward));

        // rotation matrix -> quat

        f32 m00 = right.x, m01 = up.x, m02 = -forward.x;
        f32 m10 = right.y, m11 = up.y, m12 = -forward.y;
        f32 m20 = right.z, m21 = up.z, m22 = -forward.z;

        f32 trace = m00 + m11 + m22;

        if (trace > 0.0f) {
            f32 s = 2.0f * mps::sqrtf(trace + 1.0f);
            f32 s_inv = 1.0f / s;

            quat = vec4f{
                (m21 - m12) * s_inv,
                (m02 - m20) * s_inv,
                (m10 - m01) * s_inv,
                0.25f * s
            };
        } else if (m00 > m11 && m00 > m22) {
            f32 s = 2.0f * mps::sqrtf(1.0f + m00 - m11 - m22);
            f32 s_inv = 1.0f / s;

            quat = vec4f{
                0.25f * s,
                (m01 + m10) * s_inv,
                (m02 + m20) * s_inv,
                (m21 - m12) * s_inv
            };
        } else if (m11 > m22) {
            f32 s = 2.0f * mps::sqrtf(1.0f + m11 - m00 - m22);
            f32 s_inv = 1.0f / s;

            quat = vec4f{
                (m01 + m10) * s_inv,
                0.25f * s,
                (m12 + m21) * s_inv,
                (m02 - m20) * s_inv
            };
        } else {
            f32 s = 2.0f * mps::sqrtf(1.0f + m22 - m00 - m11);
            f32 s_inv = 1.0f / s;

            quat = vec4f{
                (m02 + m20) * s_inv,
                (m12 + m21) * s_inv,
                0.25f * s,
                (m10 - m01) * s_inv
            };
        }


    }
};

}