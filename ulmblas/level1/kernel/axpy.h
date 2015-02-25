#ifndef ULMBLAS_LEVEL1_KERNEL_AXPY_H
#define ULMBLAS_LEVEL1_KERNEL_AXPY_H 1

#include <ulmblas/config/simd.h>

#if defined(USE_SSE)
#   define  SELECT_AXPY_KERNEL     sse
#   include <ulmblas/level1/kernel/sse/axpy.h>
#else
#   define  SELECT_AXPY_KERNEL     ref
#   include <ulmblas/level1/kernel/ref/axpy.h>
#endif

#endif // ULMBLAS_LEVEL1_KERNEL_AXPY_H

