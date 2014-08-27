#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_DOTAXPY_H
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_DOTAXPY_H 1

#include <src/config/simd.h>

#if defined(HAVE_SSE)
#   define  SELECT_DOTAXPY_KERNEL     sse
#   include <src/level1extensions/kernel/sse/dotaxpy.h>
#else
#   define  SELECT_DOTAXPY_KERNEL     ref
#   include <src/level1extensions/kernel/ref/dotaxpy.h>
#endif

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_DOTAXPY_H

