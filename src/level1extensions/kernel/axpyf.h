#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_AXPYF_H
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_AXPYF_H 1

#include <src/config/simd.h>

#if defined(HAVE_SSE)
#   define  SELECT_AXPYF_KERNEL     sse
#   include <src/level1extensions/kernel/sse/axpyf.h>
#else
#   define  SELECT_AXPYF_KERNEL     ref
#   include <src/level1extensions/kernel/ref/axpyf.h>
#endif

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_AXPYF_H

