#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_DOT2V_H
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_DOT2V_H 1

#include <src/config/simd.h>

#if defined(HAVE_SSE)
#   define  SELECT_DOT2V_KERNEL     sse
#   include <src/level1extensions/kernel/sse/dot2v.h>
#else
#   define  SELECT_DOT2V_KERNEL     ref
#   include <src/level1extensions/kernel/ref/dot2v.h>
#endif

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_DOT2V_H

