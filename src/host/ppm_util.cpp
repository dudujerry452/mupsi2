#include "host/ppm_util.h"

#include <iostream> 
#include <fstream> 

namespace mps::host {

  MPHD MPINL RGB_8Bit radiance_to_8bit(RGB_8Bit rgb) {

    
  }

  MPH void framebuffer_radiance_to_8bit(const FrameBuffer<RGB_8Bit>& fb, FrameBuffer<RGB_Radiance>& ret) {




  }


  MPH void save_ppm(const std::string& filename, const FrameBuffer<RGB_8Bit>& fb) {

    std::ofstream ofs(filename, std::ios::binary); 

    ofs << "P6\n" << fb.width << " " << fb.height << "\n255\n"; 

    ofs.write(reinterpret_cast<const char *>(fb.pixels), fb.bytesize()); 

    ofs.close(); 
  }

}