#ifndef ULMBLAS_SRC_LEVEL1_KERNEL_DOT_H
#define ULMBLAS_SRC_LEVEL1_KERNEL_DOT_H 1

#include <src/config/simd.h>

#if defined(HAVE_SSE)
#   define  SELECT_DOT_KERNEL      sse
#   include <src/level1/kernel/sse/dot.h>
#else
#   define  SELECT_DOT_KERNEL      ref
#   include <src/level1/kernel/ref/dot.h>
#endif

#endif // ULMBLAS_SRC_LEVEL1_KERNEL_DOT_H

