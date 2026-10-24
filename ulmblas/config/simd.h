#ifndef ULMBLAS_CONFIG_SIMD_H
#define ULMBLAS_CONFIG_SIMD_H 1

#ifndef USE_TESTPARAM

//
//  The SSE kernels need SSE3 and contain x86-64 inline assembly (64-bit
//  registers and pointers): not for 32-bit x86 and not for the x32 ABI.
//
    #if defined(__SSE3__) && defined(__x86_64__) && !defined(__ILP32__)
    #   define USE_SSE
    #endif

#endif

# endif // ULMBLAS_CONFIG_SIMD_H
