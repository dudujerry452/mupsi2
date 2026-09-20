#include <iostream>
#include <vector>
#include "core/dispatch.h"

int main() {
    const mps::i32 n = 8;
    std::vector<mps::f32> x(n, 2.0f), y(n, 3.0f);
    mps::SaxpyDispatch<mps::Backend::CPU>::run(2.0f, x.data(), y.data(), n);
    std::cout << "y[0] = " << y[0] << " (期望 7)" << std::endl;
    return 0;
}
