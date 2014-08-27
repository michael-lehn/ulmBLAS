#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_DOTXF_H
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_DOTXF_H 1

#include <src/config/simd.h>

#if defined(HAVE_SSE)
#   define  SELECT_DOTXF_KERNEL     sse
#   include <src/level1extensions/kernel/sse/dotxf.h>
#else
#   define  SELECT_DOTXF_KERNEL     ref
#   include <src/level1extensions/kernel/ref/dotxf.h>
#endif

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_DOTXF_H

