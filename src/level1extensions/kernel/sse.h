#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_SSE_H
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_SSE_H 1

#include <type_traits>

namespace ulmBLAS {

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

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_SSE_H

#include <src/level1extensions/kernel/sse.tcc>
