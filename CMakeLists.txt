cmake_minimum_required(VERSION 3.20)
project(mupsi2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ——— 依赖 ———
find_package(nlohmann_json REQUIRED)
find_package(OpenMP REQUIRED)

# ——— CPU 后端库(始终编译)———
file(GLOB CPU_SOURCES src/backend/cpu/*.cpp)
add_library(mupsi_backend_cpu STATIC ${CPU_SOURCES})
target_include_directories(mupsi_backend_cpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mupsi_backend_cpu PUBLIC OpenMP::OpenMP_CXX)

# ——— CUDA 后端库(检测到 nvcc 才编译)———
option(MUPSI_ENABLE_CUDA "Build the CUDA backend" ON)
set(MUPSI_HAS_CUDA FALSE)

if(MUPSI_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit QUIET)

    file(GLOB CUDA_SOURCES src/backend/cuda/*.cu)
    add_library(mupsi_backend_cuda STATIC ${CUDA_SOURCES})
    target_include_directories(mupsi_backend_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
    if(TARGET CUDA::cublas)
      target_link_libraries(mupsi_backend_cuda PUBLIC CUDA::cublas)
    endif()
    if(TARGET CUDA::cudart)
      target_link_libraries(mupsi_backend_cuda PUBLIC CUDA::cudart)
    endif()

    set(MUPSI_HAS_CUDA TRUE)
    message(STATUS "CUDA backend: ENABLED (${CMAKE_CUDA_COMPILER})")
  else()
    message(STATUS "CUDA backend: DISABLED (nvcc not found)")
  endif()
else()
  message(STATUS "CUDA backend: DISABLED (MUPSI_ENABLE_CUDA=OFF)")
endif()

# ——— 可执行文件:链接所有可用后端 ———
add_executable(mupsi src/main.cpp)
target_include_directories(mupsi PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mupsi PRIVATE mupsi_backend_cpu nlohmann_json::nlohmann_json)
if(MUPSI_HAS_CUDA)
  target_link_libraries(mupsi PRIVATE mupsi_backend_cuda)
  target_compile_definitions(mupsi PRIVATE MUPSI_USE_CUDA=1)
endif()
