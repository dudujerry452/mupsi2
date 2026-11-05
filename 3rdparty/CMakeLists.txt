# 单头文件第三方库。
# 头文件统一以 "3rdparty/xxx.h" 引用, 所以这里只暴露项目根目录。
# 使用时:
#   #include "3rdparty/tiny_obj_loader.h"   // 任意文件
#   #define TINYOBJLOADER_IMPLEMENTATION      // 仅一个 .cpp 里定义, 生成实现
#   #include "3rdparty/tiny_obj_loader.h"

add_library(tinyobjloader INTERFACE)
# 第三方头文件标记为 SYSTEM, 编译时不报它们的告警。
# (项目自己的头文件在 src/ 下, 仍走普通 -I, 不受影响)
target_include_directories(tinyobjloader SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
