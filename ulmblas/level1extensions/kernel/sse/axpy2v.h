#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_AXPY2V_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_AXPY2V_H 1

#include <ulmblas/level1extensions/kernel/ref/axpy2v.h>

namespace ulmBLAS { namespace sse {

using ref::axpy2v;

template <typename IndexType>
    void
    axpy2v(IndexType      n,
           const double   &alpha0,
           const double   &alpha1,
           const double   *x0,
           IndexType      incX0,
           const double   *x1,
           IndexType      incX1,
           double         *y,
           IndexType      incY);

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_AXPY2V_H 1

#include <ulmblas/level1extensions/kernel/sse/axpy2v.tcc>
