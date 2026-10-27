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
        quat = quat_from_matrix(right, up, forward); 
    }
};

}