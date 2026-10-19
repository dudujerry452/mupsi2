#include "host/ppm_util.h"
#include "core/math/math_util.h"

#include <iostream> 
#include <fstream> 

namespace mps::host {

  MPH MPINL RGB_8Bit radiance_to_8bit(RGB_F32 rgb) {
          auto encode = [](f32 x) -> byte {
              x = x > 0.0f ? x : 0.0f; // clamp
              x = x / (1.0f + x);
              x = mps::powf(x, 1.0f / 2.2f); // gamma 
              return static_cast<byte>(x * 255.0f + 0.5f); // quantitize
          };

          return {
              encode(rgb.r),
              encode(rgb.g),
              encode(rgb.b)
          };
      }

    MPH void framebuffer_radiance_to_8bit(
        const FrameBuffer<RGB_F32>& fb,
        FrameBuffer<RGB_8Bit>& ret
    ) {
        for (i32 i = 0; i < fb.width * fb.height; ++i) {
            ret.pixels[i] = radiance_to_8bit(fb.pixels[i]);
        }
    }


  MPH void save_ppm(const std::string& filename, const FrameBuffer<RGB_8Bit>& fb) {

    std::ofstream ofs(filename, std::ios::binary); 

    ofs << "P6\n" << fb.width << " " << fb.height << "\n255\n"; 

    ofs.write(reinterpret_cast<const char *>(fb.pixels), fb.bytesize()); 

    ofs.close(); 
  }

}