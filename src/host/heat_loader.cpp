#include "host/heat_loader.h"

#include <cstdio>

#include "core/utils/log.h"

namespace mps::host {

  namespace {
    template <typename T>
    bool readAll(FILE* f, std::vector<T>& out, size_t n) {
      if (n == 0) return true;
      out.resize(n);
      return std::fread(out.data(), sizeof(T), n, f) == n;
    }
    template <typename T>
    bool readVal(FILE* f, T& out) { return std::fread(&out, sizeof(T), 1, f) == 1; }
  } // namespace

  bool loadHeatKernel(const std::string& path, core::SceneHost& host) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
      mplog("loadHeatKernel: cannot open {}", path);
      return false;
    }

    u32 K, N, S, numFaces;
    if (!readVal(f, K) || !readVal(f, N) || !readVal(f, S) || !readVal(f, numFaces)) {
      mplog("loadHeatKernel: bad header");
      std::fclose(f);
      return false;
    }
    if (!readVal(f, host.heatT)) { std::fclose(f); return false; }

    bool ok = readAll(f, host.heatLam, size_t(K)) &&
              readAll(f, host.heatPhi, size_t(N) * K) &&
              readAll(f, host.heatKernelColor, size_t(S)) &&
              readAll(f, host.heatKernelTau, size_t(S)) &&
              readAll(f, host.heatKernelSharp, size_t(S)) &&
              readAll(f, host.heatKernelPhi, size_t(S) * K);

    // Skip the per-kernel face ids and barycentric coords (only used during
    // optimization; the evaluation needs kernelPhi, which is precomputed).
    if (ok) std::fseek(f, long(size_t(S) * (sizeof(u32) + 3 * sizeof(f32))), SEEK_CUR);
    if (ok) ok = readVal(f, host.heatMeanColor);
    if (ok) ok = readVal(f, host.heatBhSigma);

    host.heatN = N; host.heatK = K; host.heatS = S;
    std::fclose(f);

    if (ok)
      mplog("loadHeatKernel: {} loaded (N={} K={} S={} t={})", path, N, K, S, host.heatT);
    else
      mplog("loadHeatKernel: {} truncated", path);
    return ok;
  }

}
