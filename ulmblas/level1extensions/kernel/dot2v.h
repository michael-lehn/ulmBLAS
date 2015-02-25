#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_DOT2V_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_DOT2V_H 1

#include <ulmblas/config/simd.h>

#if defined(USE_SSE)
#   define  SELECT_DOT2V_KERNEL     sse
#   include <ulmblas/level1extensions/kernel/sse/dot2v.h>
#else
#   define  SELECT_DOT2V_KERNEL     ref
#   include <ulmblas/level1extensions/kernel/ref/dot2v.h>
#endif

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_DOT2V_H

