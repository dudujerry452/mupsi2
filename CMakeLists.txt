cmake_minimum_required(VERSION 3.20)
project(mupsi2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ——— 依赖(CPU 系列)———
find_package(nlohmann_json REQUIRED)

# ——— CPU 核心:无条件编译,任何机器都能过 ———
add_executable(mupsi src/main.cpp)
target_link_libraries(mupsi PRIVATE nlohmann_json::nlohmann_json)

# ——— CUDA 后端:检测到 nvcc 才编译,否则静默跳过 ———
option(MUPSI_ENABLE_CUDA "Build the CUDA backend" ON)

if(MUPSI_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # cuBLAS(CUDA::cublas 由 CUDAToolkit 提供;若 nix 布局下找不到,
    # 改为直接 link ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} 中的 cublas)
    find_package(CUDAToolkit QUIET)

    add_library(mupsi_backend_cuda STATIC src/backend/cuda/hello.cu)
    target_include_directories(mupsi_backend_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(mupsi_backend_cuda PRIVATE MUPSI_ENABLE_CUDA=1)
    if(TARGET CUDA::cublas)
      target_link_libraries(mupsi_backend_cuda PRIVATE CUDA::cublas)
    endif()

    message(STATUS "CUDA backend: ENABLED (${CMAKE_CUDA_COMPILER})")
  else()
    set(MUPSI_ENABLE_CUDA OFF)
    message(STATUS "CUDA backend: DISABLED (nvcc not found)")
  endif()
else()
  message(STATUS "CUDA backend: DISABLED (MUPSI_ENABLE_CUDA=OFF)")
endif()
