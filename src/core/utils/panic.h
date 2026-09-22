#pragma once 

#ifdef __CUDA_ARCH__
    #define lpanic(fmt, ...) \
        do { \
            printf("[GPU FATAL] " fmt "\n", ##__VA_ARGS__); \
            __trap(); \
        } while(0)
#else
    #include <cstdlib>
    #define lpanic(fmt, ...) \
        do { \
            std::fprintf(stderr, "[CPU FATAL] " fmt "\n", ##__VA_ARGS__); \
            std::abort(); \
        } while(0)
#endif