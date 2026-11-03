#include "src/host/img_loader.h"
  
#include <cmath>
#include <stdexcept>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "3rdparty/stb_image.h"

namespace mps::host {


  MPH Texture<RGB_F32> load_albedo(const char* path) {
    i32 width, height, channels;

    stbi_uc* src = stbi_load(
        path,
        &width,
        &height,
        &channels,
        3
    );

    if (!src) {
        throw std::runtime_error(
            std::string("Failed to load image: ") +
            stbi_failure_reason()
        );
    }

    auto srgb_to_linear = [](f32 x) -> f32 {
        return x <= 0.04045f
            ? x / 12.92f
            : std::pow((x + 0.055f) / 1.055f, 2.4f);
    };

    Texture<RGB_F32> tex{};
    tex.width = width;
    tex.height = height;
    tex.pixels = new RGB_F32[width * height];

    for (i32 i = 0; i < width * height; ++i) {
        f32 r = src[i * 3 + 0] / 255.0f;
        f32 g = src[i * 3 + 1] / 255.0f;
        f32 b = src[i * 3 + 2] / 255.0f;

        tex.pixels[i] = RGB_F32{
            srgb_to_linear(r),
            srgb_to_linear(g),
            srgb_to_linear(b)
        };
    }

    stbi_image_free(src);

    return tex;
}

  
}