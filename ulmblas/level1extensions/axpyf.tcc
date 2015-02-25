#ifndef ULMBLAS_LEVEL1EXTENSIONS_AXPYF_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_AXPYF_TCC 1

#include <ulmblas/level1extensions/axpyf.h>
#include <ulmblas/level1extensions/kernel/axpyf.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename VA, typename VX,
          typename VY>
void
axpyf(IndexType      n,
      const Alpha    &alpha,
      const VA       *a,
      IndexType      incA,
      const VX       *X,
      IndexType      incRowX,
      IndexType      incColX,
      VY             *y,
      IndexType      incY)
{
    SELECT_AXPYF_KERNEL::axpyf(n, alpha, a, incA, X, incRowX, incColX, y, incY);
}

template <typename IndexType, typename Alpha, typename VA, typename VX,
          typename VY>
void
acxpyf(IndexType      n,
       const Alpha    &alpha,
       const VA       *a,
       IndexType      incA,
       const VX       *X,
       IndexType      incRowX,
       IndexType      incColX,
       VY             *y,
       IndexType      incY)
{
    SELECT_AXPYF_KERNEL::acxpyf(n, alpha,
                                a, incA,
                                X, incRowX, incColX,
                                y, incY);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_AXPYF_TCC
