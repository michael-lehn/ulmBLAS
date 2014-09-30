#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXF_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXF_H 1

#include <ulmblas/level1extensions/kernel/ref/dotxf.h>

namespace ulmBLAS { namespace sse {

using ref::dotuxf;

template <typename T>
    int
    dotuxf_fusefactor();

template <typename IndexType>
    void
    dotuxf(IndexType      n,
           const double   *X,
           IndexType      incRowX,
           IndexType      incColX,
           const double   *y,
           IndexType      incY,
           double         *result,
           IndexType      resultInc);

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXF_H

#include <ulmblas/level1extensions/kernel/sse/dotxf.tcc>
