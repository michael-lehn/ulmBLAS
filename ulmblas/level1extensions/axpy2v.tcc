#ifndef ULMBLAS_LEVEL1EXTENSIONS_AXPY2V_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_AXPY2V_TCC 1

#include <ulmblas/level1extensions/axpy2v.h>
#include <ulmblas/level1extensions/kernel/axpy2v.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha0, typename Alpha1,
          typename VX0, typename VX1, typename VY>
void
axpy2v(IndexType      n,
       const Alpha0   &alpha0,
       const Alpha1   &alpha1,
       const VX0      *x0,
       IndexType      incX0,
       const VX1      *x1,
       IndexType      incX1,
       VY             *y,
       IndexType      incY)
{
    SELECT_AXPY2V_KERNEL::axpy2v(n, alpha0, alpha1,
                                 x0, incX0, x1, incX1,
                                 y, incY);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_AXPY2V_TCC 1
