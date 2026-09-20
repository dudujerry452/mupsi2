#include "core/test.h"

namespace mps {

// CPU 特化定义:OpenMP 循环调用内核
void SaxpyDispatch<Backend::CPU>::run(f32 a, const f32* x, f32* y, i32 n) {
    #pragma omp parallel for
    for (i32 i = 0; i < n; i++) saxpy(a, x[i], y[i]);
}

}  // namespace mps
