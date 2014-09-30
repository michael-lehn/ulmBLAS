#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_AXPYF_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_AXPYF_H

#include <ulmblas/level1extensions/kernel/ref/axpyf.h>

namespace ulmBLAS { namespace sse {

using ref::axpyf;

template <typename T>
    int
    axpyf_fusefactor();

template <typename IndexType>
    void
    axpyf(IndexType      n,
          const double   &alpha,
          const double   *a,
          IndexType      incA,
          const double   *x,
          IndexType      incRowX,
          IndexType      incColX,
          double         *y,
          IndexType      incY);

} } // namespace sse, ulmBLAS

#include <ulmblas/level1extensions/kernel/sse/axpyf.tcc>

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_AXPYF_H
