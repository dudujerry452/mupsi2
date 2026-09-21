#include <iostream>
#include <vector>

#include "core/core.inl"
#include "kernels/test/dispatch.h"

#if defined(MUPSI_USE_CUDA)
constexpr mps::Backend ACTIVE_BACKEND = mps::Backend::CUDA;
#else
constexpr mps::Backend ACTIVE_BACKEND = mps::Backend::CPU;
#endif

int main() {

    std::vector<mps::u32> out(100);
    mps::kernels::test_func<ACTIVE_BACKEND>::run(100, out.data()); 

    for(auto x: out) {
        std::cout << x << " "; 
    }
    std::cout << std::endl; 

    return 0;
}
