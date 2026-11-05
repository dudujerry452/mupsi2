# 3rdparty

单头文件第三方库, 直接以 `#include "3rdparty/..."` 引用 (项目根目录在 include path 中)。

| 文件 | 版本 | 来源 |
| --- | --- | --- |
| `stb_image.h` | v2.30 | https://github.com/nothings/stb |
| `tiny_obj_loader.h` | v2.0.0 (release `45636bd`) | https://github.com/tinyobjloader/tinyobjloader |

## OBJ 加载用法

实现只在一个 `.cpp` 中生成 (和 `stb_image` 一样), 其他文件只 include:

```cpp
// mesh_loader.cpp
#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdparty/tiny_obj_loader.h"
```

```cpp
// 任意其他文件
#include "3rdparty/tiny_obj_loader.h"

tinyobj::ObjReader reader;
tinyobj::ObjReaderConfig config;
config.triangulate = true;
if (!reader.ParseFromFile("model.obj", config)) {
  // reader.Error()
}
auto& attrib = reader.GetAttrib();   // positions / normals / texcoords
auto& shapes = reader.GetShapes();   // 每个 shape 有 mesh 和 material_id
auto& materials = reader.GetMaterials();
```

CMake 里已由 `mupsi_header` 传递 `tinyobjloader` 这个 INTERFACE target,
凡是链接了 `mupsi_header` 的目标都能直接 include, 无需额外配置。
