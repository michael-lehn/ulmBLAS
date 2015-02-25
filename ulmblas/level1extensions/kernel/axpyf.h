#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_AXPYF_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_AXPYF_H 1

#include <ulmblas/config/simd.h>

#if defined(USE_SSE)
#   define  SELECT_AXPYF_KERNEL     sse
#   include <ulmblas/level1extensions/kernel/sse/axpyf.h>
#else
#   define  SELECT_AXPYF_KERNEL     ref
#   include <ulmblas/level1extensions/kernel/ref/axpyf.h>
#endif

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_AXPYF_H

