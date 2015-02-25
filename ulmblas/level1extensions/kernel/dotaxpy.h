#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_DOTAXPY_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_DOTAXPY_H 1

#include <ulmblas/config/simd.h>

#if defined(USE_SSE)
#   define  SELECT_DOTAXPY_KERNEL     sse
#   include <ulmblas/level1extensions/kernel/sse/dotaxpy.h>
#else
#   define  SELECT_DOTAXPY_KERNEL     ref
#   include <ulmblas/level1extensions/kernel/ref/dotaxpy.h>
#endif

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_DOTAXPY_H
