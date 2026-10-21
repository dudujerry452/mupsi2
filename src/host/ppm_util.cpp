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
    ofs.write(reinterpret_cast<const char*>(fb.pixels), fb.bytesize());
    ofs.close();
  }

}
