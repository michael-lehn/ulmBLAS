#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOT2V_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOT2V_H 1

#include <ulmblas/level1extensions/kernel/ref/dot2v.h>

namespace ulmBLAS { namespace sse {

using ref::dotu2v;

template <typename IndexType>
    void
    dotu2v(IndexType      n,
           const double   *x0,
           IndexType      incX0,
           const double   *x1,
           IndexType      incX1,
           double         *y,
           IndexType      incY,
           double         *result,
           IndexType      resultInc);

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOT2V_H 1

#include <ulmblas/level1extensions/kernel/sse/dot2v.tcc>
