#pragma once 

#include "core/basic/types.h"
#include "core/tracer/trace.h"

namespace mps::host {

  // modify orig, draw lines
  MPH void draw_debug(Texture<RGB_F32> orig, Texture<RGB_F32> depth, const core::debug_render_lines& debug);

}