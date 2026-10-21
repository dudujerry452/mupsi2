#include "host/hdr_loader.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "core/utils/log.h"

namespace mps::host {

  namespace {

    // Convert one RGBE pixel to linear float RGB.
    static RGB_F32 rgbeToRgb(const unsigned char* p) {
      if (p[3] == 0) return {0.0f, 0.0f, 0.0f};
      const f32 f = std::ldexp(1.0f, i32(p[3]) - (128 + 8));
      return {f32(p[0]) * f, f32(p[1]) * f, f32(p[2]) * f};
    }

    // Decode a single RLE run (shared by R/G/B/E channels).
    bool decodeRun(FILE* f, unsigned char* dst, i32 width) {
      i32 x = 0;
      while (x < width) {
        int count = std::fgetc(f);
        if (count == EOF) return false;
        if (count > 128) {
          count -= 128;
          int value = std::fgetc(f);
          if (value == EOF) return false;
          while (count-- > 0 && x < width) dst[x++] = (unsigned char)value;
        } else {
          while (count-- > 0 && x < width) {
            int value = std::fgetc(f);
            if (value == EOF) return false;
            dst[x++] = (unsigned char)value;
          }
        }
      }
      return true;
    }

  } // namespace

  HdrImage loadHdr(const std::string& path) {
    HdrImage img;

    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
      mplog("loadHdr: cannot open {}", path);
      return img;
    }

    // ——— header ———
    char line[256];
    if (!std::fgets(line, sizeof(line), f) || std::strncmp(line, "#?", 2) != 0) {
      mplog("loadHdr: {} is not a Radiance file", path);
      std::fclose(f);
      return img;
    }

    // Skip comment/format lines; stop at the resolution line (after a blank).
    i32 width = 0, height = 0;
    while (std::fgets(line, sizeof(line), f)) {
      if (line[0] == '\n') continue;          // blank
      if (line[0] == '#') continue;           // comment
      if (std::strstr(line, "FORMAT=")) continue;
      // Resolution line: e.g. "-Y 512 +X 1024"
      char axis1[3] = {0, 0, 0}, axis2[3] = {0, 0, 0}; i32 v1 = 0, v2 = 0;
      if (std::sscanf(line, "%2s %d %2s %d", axis1, &v1, axis2, &v2) == 4) {
        if (axis1[1] == 'Y') { height = v1; } else if (axis1[1] == 'X') { width = v1; }
        if (axis2[1] == 'Y') { height = v2; } else if (axis2[1] == 'X') { width = v2; }
        break;
      }
    }

    if (width <= 0 || height <= 0 || width > 65535) {
      mplog("loadHdr: bad resolution {}x{} in {}", width, height, path);
      std::fclose(f);
      return img;
    }

    img.width = width;
    img.height = height;
    img.pixels.resize(size_t(width) * size_t(height));

    // ——— scanlines ———
    std::vector<unsigned char> scan(width * 4);
    std::vector<unsigned char> r(width), g(width), b(width), e(width);

    for (i32 y = 0; y < height; ++y) {
      // Peek the first 4 bytes to detect RLE vs flat.
      unsigned char head[4];
      size_t got = std::fread(head, 1, 4, f);
      if (got != 4) { mplog("loadHdr: truncated at scanline {}", y); break; }

      bool rle = (head[0] == 0x02 && head[1] == 0x02 &&
                  int(head[2]) * 256 + int(head[3]) == width);

      if (rle) {
        if (!decodeRun(f, r.data(), width) || !decodeRun(f, g.data(), width) ||
            !decodeRun(f, b.data(), width) || !decodeRun(f, e.data(), width)) {
          mplog("loadHdr: bad RLE at scanline {}", y);
          break;
        }
      } else {
        // Flat: the 4 head bytes belong to the first pixel.
        r[0] = head[0]; g[0] = head[1]; b[0] = head[2]; e[0] = head[3];
        if (std::fread(scan.data() + 4, 1, size_t(width) * 4 - 4, f) != size_t(width) * 4 - 4) {
          mplog("loadHdr: truncated flat scanline {}", y);
          break;
        }
        for (i32 x = 1; x < width; ++x) {
          r[x] = scan[4 * x + 0]; g[x] = scan[4 * x + 1];
          b[x] = scan[4 * x + 2]; e[x] = scan[4 * x + 3];
        }
      }

      for (i32 x = 0; x < width; ++x) {
        unsigned char px[4] = {r[x], g[x], b[x], e[x]};
        img.pixels[size_t(y) * width + x] = rgbeToRgb(px);
      }
    }

    std::fclose(f);
    mplog("loadHdr: {} {}x{} loaded", path, width, height);
    return img;
  }

}
