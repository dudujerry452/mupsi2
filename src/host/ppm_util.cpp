#include "host/ppm_util.h"

#include <fstream>

namespace mps::host {

  MPHD void framebuffer_radiance_to_8bit(const FrameBuffer<RGB_F32>& fb, FrameBuffer<RGB_8Bit>& ret) {
    for (i32 i = 0; i < fb.width * fb.height; ++i) {
      ret.pixels[i] = radiance_to_8bit(fb.pixels[i]);
    }
  }

  MPH void save_ppm(const std::string& filename, const FrameBuffer<RGB_8Bit>& fb) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P6\n" << fb.width << " " << fb.height << "\n255\n";
    // Framebuffer convention: (0,0) is BOTTOM-left (row 0 = bottom row), and the
    // PPM is written top-down. Emit rows in reverse so the bottom-left of the
    // framebuffer lands at the bottom-left of the image.
    for (i32 y = fb.height - 1; y >= 0; --y) {
      ofs.write(reinterpret_cast<const char*>(fb.pixels + size_t(y) * fb.width),
                fb.width * sizeof(RGB_8Bit));
    }
    ofs.close();
  }

}
