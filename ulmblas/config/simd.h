#ifndef ULMBLAS_CONFIG_SIMD_H
#define ULMBLAS_CONFIG_SIMD_H 1

//
//  The test configurations with small block sizes (USE_TESTPARAM) use the
//  reference kernels, unless USE_TESTPARAM_KEEP_KERNELS is defined.
//
#if !defined(USE_TESTPARAM) || defined(USE_TESTPARAM_KEEP_KERNELS)

//
//  The SSE kernels need SSE3 and contain x86-64 inline assembly (64-bit
//  registers and pointers): not for 32-bit x86 and not for the x32 ABI.
//
    #if defined(__SSE3__) && defined(__x86_64__) && !defined(__ILP32__)
    #   define USE_SSE
    //
    //  The AVX micro kernel needs AVX and FMA (e.g. ARCHFLAGS='-mavx -mfma').
    //
    #   if defined(__AVX__) && defined(__FMA__)
    #       define USE_AVX
    #   endif
    #endif

#endif

# endif // ULMBLAS_CONFIG_SIMD_H
