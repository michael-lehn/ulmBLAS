#ifndef ULMBLAS_CONFIG_SIMD_H
#define ULMBLAS_CONFIG_SIMD_H 1

#ifndef USE_TESTPARAM

    #if defined(__SSE3__)
    #   define USE_SSE
    #endif

#endif

# endif // ULMBLAS_CONFIG_SIMD_H
