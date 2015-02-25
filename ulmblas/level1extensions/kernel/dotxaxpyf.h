#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_DOTXAXPYF_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_DOTXAXPYF_H 1

#include <ulmblas/config/simd.h>

#if defined(USE_SSE)
#   define  SELECT_DOTXAXPYF_KERNEL     sse
#   include <ulmblas/level1extensions/kernel/sse/dotxaxpyf.h>
#else
#   define  SELECT_DOTXAXPYF_KERNEL     ref
#   include <ulmblas/level1extensions/kernel/ref/dotxaxpyf.h>
#endif

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_DOTXAXPYF_H
