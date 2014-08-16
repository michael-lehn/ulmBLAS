#ifndef ULMBLAS_SRC_LEVEL3_KERNEL_KERNEL_H
#define ULMBLAS_SRC_LEVEL3_KERNEL_KERNEL_H 1

#include <src/config/simd.h>

#if defined(USE_SSE)
#   include <src/level3/kernel/sse.h>
#   include <src/level3/kernel/sse.tcc>
#endif


#endif // ULMBLAS_SRC_LEVEL3_KERNEL_KERNEL_H

