{
  description = "mupsi2 — GPIS renderer (swappable CPU/CUDA backends)";

  inputs = {
    # 锁定到与 ~/nixos 相同的 nixpkgs rev,复用其已缓存的 CUDA 包,
    # 避免在 WSL2 上重复下载数 GB 的 CUDA 工具链。与 ~/nixos/flake.lock 保持一致。
    nixpkgs.url = "github:NixOS/nixpkgs/4bd9165a9165d7b5e33ae57f3eecbcb28fb231c9";
  };

  outputs = { self, nixpkgs }:
    let
      # 当前两台机器均为 x86_64-linux。若要跨平台,改成遍历
      # ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"]。
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;   # CUDA 是 unfree 包
      };
      cuda = pkgs.cudaPackages;

      # 两套配置共用的基础编译工具(CPU/CUDA 都要)
      base = with pkgs; [
        cmake
        ninja
        pkg-config
        gnumake
        gdb
        nlohmann_json   # config 解析
      ];

      # CUDA 专属依赖:编译器 + runtime + cuBLAS + CUTLASS
      cudaDeps = [
        cuda.cuda_nvcc       # nvcc 编译器
        cuda.cuda_cudart     # CUDA runtime
        cuda.libcublas       # cuBLAS
        cuda.cutlass         # CUTLASS(头文件模板库)
      ];
    in
    {
      devShells.${system} = {
        # 默认 = CPU:`nix develop` 直接进,无需 GPU、无需 Eigen
        default = pkgs.mkShell {
          packages = base;
          shellHook = ''
            echo "=== mupsi2 dev shell [CPU] ==="
            echo "gcc $(gcc --version | head -1 | cut -d' ' -f4)"
            echo "CUDA backend: disabled"
          '';
        };

        # 显式 CPU 配置,等价于 default
        cpu = pkgs.mkShell {
          packages = base;
          shellHook = ''
            echo "=== mupsi2 dev shell [CPU] ==="
            echo "gcc $(gcc --version | head -1 | cut -d' ' -f4)"
          '';
        };

        # CUDA 配置:`nix develop .#cuda`
        cuda = pkgs.mkShell {
          packages = base ++ cudaDeps;
          shellHook = ''
            echo "=== mupsi2 dev shell [CUDA] ==="
            echo "nvcc $(nvcc --version | tail -1)"
            echo "cuBLAS + CUTLASS: available"
            export CUDA_PATH=${cuda.cuda_nvcc}
          '';
        };
      };
    };
}
