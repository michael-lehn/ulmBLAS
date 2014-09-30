#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_AXPY2V_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_AXPY2V_H 1

#include <ulmblas/config/simd.h>

#if defined(HAVE_SSE)
#   define  SELECT_AXPY2V_KERNEL     sse
#   include <ulmblas/level1extensions/kernel/sse/axpy2v.h>
#else
#   define  SELECT_AXPY2V_KERNEL     ref
#   include <ulmblas/level1extensions/kernel/ref/axpy2v.h>
#endif

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_AXPY2V_H

