#ifndef ULMBLAS_SRC_LEVEL1_KERNEL_AXPY_H
#define ULMBLAS_SRC_LEVEL1_KERNEL_AXPY_H 1

#include <src/config/simd.h>

#if defined(HAVE_SSE)
#   define  SELECT_AXPY_KERNEL     sse
#   include <src/level1/kernel/sse/axpy.h>
#else
#   define  SELECT_AXPY_KERNEL     ref
#   include <src/level1/kernel/ref/axpy.h>
#endif

#endif // ULMBLAS_SRC_LEVEL1_KERNEL_AXPY_H

