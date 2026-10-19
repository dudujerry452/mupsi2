#pragma once 

#include "core/basic/types.h"
#include "core/scene/scene.h"
#include "core/tracer/intersect.h"

namespace mps::core {

  MPHD RGB_F32 trace(const Ray ray, const SceneDevice scene); 

}