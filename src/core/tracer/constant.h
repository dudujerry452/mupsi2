#pragma once

#include "core/basic/types.h"

namespace mps::core {

  // ——— 固定量(绝对长度,不随场景尺寸缩放)———
  // 命中表面后沿法线偏移 / 反弹后 ray.min —— 用于避免重复撞击同一表面,
  // 是"偏移一点点即可"的绝对量,与物体多大无关。
  constexpr f32 EPS_SURFACE = 1e-3f;
  constexpr f32 EPS_NEAR    = 1e-2f;

  // ——— 随场景尺寸缩放的量(使用处乘 medium.sceneScale)———
  // 场景特征长度 sceneScale(≈物体半径)。这些是"比例",乘 sceneScale 得到世界单位。

  // marching 最大步长 = sceneScale / 300(场景越大步长越大)。
  constexpr f32 MARCH_MAX_DT_FRAC = 1.0f / 300.0f;
  // marching 最远距离(安全上限)。
  constexpr f32 MARCH_FAR_CLAMP_FRAC = 20.0f;
  // 符号翻转判定的最小 |Δf|(f 的量级随场景缩放,阈值也随场景缩放)。
  constexpr f32 MARCH_SIGN_EPS_FRAC = 1.0e-8f;
  // SDF sphere tracing 最小步长。
  constexpr f32 SDF_MIN_STEP_FRAC = 1.0e-5f;

  // ——— 无量纲量(不缩放)———
  constexpr i32 MARCH_MIN_DEPTH = 100;  // dt 被最大步长截断时的步数下限
  constexpr i32 SDF_MAX_ITER    = 512;  // SDF sphere tracing 最大迭代
  constexpr i32 SDF_BISECT      = 16;   // 符号翻转后二分精化步数

}
