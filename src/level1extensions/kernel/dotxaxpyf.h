#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_DOTXAXPYF_H
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_DOTXAXPYF_H 1

#include <src/config/simd.h>

#if defined(HAVE_SSE)
#   define  SELECT_DOTXAXPYF_KERNEL     sse
#   include <src/level1extensions/kernel/sse/dotxaxpyf.h>
#else
#   define  SELECT_DOTXAXPYF_KERNEL     ref
#   include <src/level1extensions/kernel/ref/dotxaxpyf.h>
#endif

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_DOTXAXPYF_H
