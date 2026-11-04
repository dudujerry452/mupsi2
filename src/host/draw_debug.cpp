#include "host/draw_debug.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace mps::host {

  namespace {

    struct CamFrame {
      vec3f wpos;
      vec3f right, up, forward;
      f32 znear, real_width, real_height;
      i32 width, height;
    };

    CamFrame make_frame(const core::Camera& cam, i32 width, i32 height) {
      CamFrame f;
      f.wpos = cam.wpos;
      f.right   = mps::quat_rotate(cam.quat, vec3f{1.0f, 0.0f, 0.0f});
      f.up      = mps::quat_rotate(cam.quat, vec3f{0.0f, 1.0f, 0.0f});
      f.forward = mps::quat_rotate(cam.quat, vec3f{0.0f, 0.0f, -1.0f});
      f.znear = cam.znear;
      f.real_height = cam.znear * mps::tanf(cam.fov * 0.5f) * 2.0f;
      f.real_width = cam.aspect * f.real_height;
      f.width = width;
      f.height = height;
      return f;
    }

    // world -> camera space: x=right, y=up, z=forward(z>0 在相机前方)
    vec3f world_to_cam(vec3f p, const CamFrame& f) {
      vec3f d = p - f.wpos;
      return vec3f{mps::dot(d, f.right), mps::dot(d, f.up), mps::dot(d, f.forward)};
    }

    // camera space -> 屏幕空间 (col, row, depth), depth = 到相机的距离
    vec3f cam_to_screen(vec3f pc, const CamFrame& f) {
      f32 inv_z = f.znear / pc.z; // pc.z > 0
      f32 x = 0.5f + pc.x * inv_z / f.real_width;
      f32 y = 0.5f + pc.y * inv_z / f.real_height;
      return vec3f{
        x * (f32)f.width - 0.5f,
        (1.0f - y) * (f32)f.height - 0.5f,
        mps::norm(pc)
      };
    }

  }

  MPH void draw_debug(Texture<RGB_F32> orig, Texture<RGB_F32> depth, const core::debug_render_lines* debug, u32 num, const core::Camera& cam) {
    CamFrame f = make_frame(cam, orig.width, orig.height);
    const f32 eps = 0.2f; // 深度比较容差

    for (u32 p = 0; p < num; ++p) {
      const auto& path = debug[p];
      // 反向画: 先画后面的 bounce, 最后画主光线, 让主光线的白点不被覆盖
      for (i32 i = (i32)path.num_bounce - 1; i >= 0; --i) {
        const auto& line = path.lines[i];

        // world 线段 -> camera space
        vec3f ac = world_to_cam(line.start, f);
        vec3f bc = world_to_cam(line.end, f);

        // 起点已经用 RAY_MIN 偏移过, 不会 /0; 防御性跳过相机后方的点
        if (ac.z <= 0.0f || bc.z <= 0.0f) continue;

        // 端点投影到屏幕
        vec3f s0 = cam_to_screen(ac, f);
        vec3f s1 = cam_to_screen(bc, f);

        i32 x0 = (i32)std::lroundf(s0.x), y0 = (i32)std::lroundf(s0.y);
        i32 x1 = (i32)std::lroundf(s1.x), y1 = (i32)std::lroundf(s1.y);

        RGB_F32 col = line.albedo;
        col.r = col.r > 1.0f ? 1.0f : (col.r < 0.0f ? 0.0f : col.r);
        col.g = col.g > 1.0f ? 1.0f : (col.g < 0.0f ? 0.0f : col.g);
        col.b = col.b > 1.0f ? 1.0f : (col.b < 0.0f ? 0.0f : col.b);

        // Bresenham 画线, 每个像素算一个深度值
        i32 dx = std::abs(x1 - x0);
        i32 dy = std::abs(y1 - y0);
        i32 sx = x0 < x1 ? 1 : -1;
        i32 sy = y0 < y1 ? 1 : -1;
        i32 err = dx - dy;
        i32 steps = std::max(dx, dy);

        i32 x = x0, y = y0;
        for (i32 s = 0; ; ++s) {
          f32 t = steps > 0 ? (f32)s / (f32)steps : 0.0f;
          vec3f cp = ac + (bc - ac) * t; // 相机空间线性插值(= 世界空间线性插值)
          f32 d = mps::norm(cp);         // 该像素处光线的深度

          if (x >= 0 && x < orig.width && y >= 0 && y < orig.height) {
            f32 dh = depth.pixel(x, y).r; // trace_depth 把 hit.t 写进 rgb 三通道
            bool occluded = dh > 0.0001f && d > dh + eps;
            if (!occluded) {
              orig.pixel(x, y) = col; // 可见才覆盖
            }
          }

          if (x == x1 && y == y1) break;
          i32 e2 = 2 * err;
          if (e2 > -dy) { err -= dy; x += sx; }
          if (e2 <  dx) { err += dx; y += sy; }
        }
      }
    }
  }

}
