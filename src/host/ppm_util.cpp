#include "host/ppm_util.h"
#include "core/basic/types.h"
#include "core/math/math_util.h"

#include <iostream> 
#include <fstream> 

namespace mps::host {

    MPH MPINL RGB_F32 bit8ToRadiance(RGB_8Bit rgb) {
        auto decode = [](byte v) -> f32 {
            f32 x = static_cast<f32>(v) * (1.0f / 255.0f); 
            return mps::powf(x, 2.2f); 
        }; 
        return RGB_F32{decode(rgb.r), decode(rgb.g), decode(rgb.b)}; 
    }

    MPH MPINL RGB_8Bit radianceTo8bit(RGB_F32 rgb) {
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

    MPH void radianceTo8bit(
        Texture<RGB_8Bit>& out,
        const Texture<RGB_F32>& fb
    ) {
        for (i32 i = 0; i < fb.width * fb.height; ++i) {
            out.pixels[i] = radianceTo8bit(fb.pixels[i]);
        }
    }


    MPH void savePPM(const std::string& filename, const Texture<RGB_8Bit>& fb) {

        std::ofstream ofs(filename, std::ios::binary); 

        ofs << "P6\n" << fb.width << " " << fb.height << "\n255\n"; 

        ofs.write(reinterpret_cast<const char *>(fb.pixels), fb.bytesize()); 

        ofs.close(); 
    }

}