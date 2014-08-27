#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_SSE_DOT2V_H
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_SSE_DOT2V_H 1

namespace ulmBLAS { namespace sse {

template <typename IndexType, typename VX0, typename VX1, typename VY,
          typename Result>
    void
    dotu2v(IndexType      n,
           const VX0      *x0,
           IndexType      incX0,
           const VX1      *x1,
           IndexType      incX1,
           VY             *y,
           IndexType      incY,
           Result         *result,
           IndexType      resultInc);

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_SSE_DOT2V_H 1

#include <src/level1extensions/kernel/sse/dot2v.tcc>
