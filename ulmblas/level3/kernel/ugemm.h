#ifndef ULMBLAS_LEVEL3_KERNEL_UGEMM_H
#define ULMBLAS_LEVEL3_KERNEL_UGEMM_H 1

#include <ulmblas/config/simd.h>

#if defined(HAVE_SSE)
#   define  SELECT_UGEMM_KERNEL     sse
#   include <ulmblas/level3/kernel/sse/ugemm.h>
#else
#   define  SELECT_UGEMM_KERNEL     ref
#   include <ulmblas/level3/kernel/ref/ugemm.h>
#endif

#endif // ULMBLAS_LEVEL3_KERNEL_UGEMM_H

