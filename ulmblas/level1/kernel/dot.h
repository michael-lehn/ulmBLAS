#ifndef ULMBLAS_LEVEL1_KERNEL_DOT_H
#define ULMBLAS_LEVEL1_KERNEL_DOT_H 1

#include <ulmblas/config/simd.h>

#if defined(USE_SSE)
#   define  SELECT_DOT_KERNEL      sse
#   include <ulmblas/level1/kernel/sse/dot.h>
#else
#   define  SELECT_DOT_KERNEL      ref
#   include <ulmblas/level1/kernel/ref/dot.h>
#endif

#endif // ULMBLAS_LEVEL1_KERNEL_DOT_H

