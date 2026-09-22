cmake_minimum_required(VERSION 3.20)
project(mupsi2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ——— 编译告警 ———
add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
)

# ——— 依赖 ———
find_package(nlohmann_json REQUIRED)
find_package(OpenMP REQUIRED)

# ——— CUDA 检测(先于 core, 决定 core 由 g++ 还是 nvcc 编译)———
option(MUPSI_ENABLE_CUDA "Build the CUDA backend" ON)
set(MUPSI_HAS_CUDA FALSE)

if(MUPSI_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA) # define CMAKE_CUDA_COMPILER
  if(CMAKE_CUDA_COMPILER) # nvcc success
    enable_language(CUDA) # enable .cu
    set(CMAKE_CUDA_ARCHITECTURES native) # tells nvcc the gpu arch
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit QUIET)

    set(MUPSI_HAS_CUDA TRUE)
    message(STATUS "CUDA backend: ENABLED (${CMAKE_CUDA_COMPILER})")
  else()
    message(STATUS "CUDA backend: DISABLED (nvcc not found)")
  endif()
else()
  message(STATUS "CUDA backend: DISABLED (MUPSI_ENABLE_CUDA=OFF)")
endif()


# ——— 头文件 ———
add_library(mupsi_header INTERFACE)
target_include_directories(mupsi_header INTERFACE ${CMAKE_SOURCE_DIR}/src)

# ——— core:设备无关计算代码; 只存在g++时用g++编译, 否则用nvcc编译(因为标记了MPHD,因此编译为device和host)———
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS src/core/*.cpp)
add_library(mupsi_core STATIC ${CORE_SOURCES})
target_link_libraries(mupsi_core PRIVATE mupsi_header)
if(MUPSI_HAS_CUDA)
  # 标记core files
  set_source_files_properties(${CORE_SOURCES} PROPERTIES 
    LANGUAGE CUDA
    COMPILE_OPTIONS "-x;cu;-dc"
  )
  # 设置core为CUDA语言, 允许跨库调用
  set_target_properties(mupsi_core PROPERTIES 
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
  )

endif()

if(NOT MUPSI_HAS_CUDA) 
# ——— CPU 后端库(始终编译:kernels/*.cpp + host/*.cpp, 仅 g++)———
  file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS src/kernels/*.cpp src/host/*.cpp)
  add_library(mupsi_backend_cpu STATIC ${CPU_SOURCES})
  target_link_libraries(mupsi_backend_cpu PUBLIC mupsi_core mupsi_header OpenMP::OpenMP_CXX)
else()
# ——— CUDA 后端库(kernels/*.cu, 检测到 nvcc 才编译)———
  file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS src/kernels/*.cu src/host/*.cpp)
  set_source_files_properties(${CORE_SOURCES} PROPERTIES 
    LANGUAGE CUDA
    COMPILE_OPTIONS "-x;cu;-dc"
  )
  add_library(mupsi_backend_cuda STATIC ${CUDA_SOURCES})
  target_link_libraries(mupsi_backend_cuda PUBLIC mupsi_core mupsi_header)
  # 允许跨库调用
  set_target_properties(mupsi_backend_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  if(TARGET CUDA::cublas)
    message(STATUS "CUDA cublas ENABLED")
    target_link_libraries(mupsi_backend_cuda PUBLIC CUDA::cublas)
  endif()
  if(TARGET CUDA::cudart)
    message(STATUS "CUDA cudart ENABLED")
    target_link_libraries(mupsi_backend_cuda PUBLIC CUDA::cudart)
  endif()
endif()

# ——— 可执行文件:链接所有可用后端 ———
add_executable(mupsi src/main.cpp)
target_link_libraries(mupsi PRIVATE mupsi_header mupsi_core nlohmann_json::nlohmann_json)
if(NOT MUPSI_HAS_CUDA)
  target_link_libraries(mupsi PRIVATE mupsi_backend_cpu)
else()
  set_target_properties(mupsi PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(mupsi PRIVATE mupsi_backend_cuda)
  target_compile_definitions(mupsi PRIVATE MUPSI_USE_CUDA=1)
endif()
