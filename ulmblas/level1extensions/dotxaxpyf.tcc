#ifndef ULMBLAS_LEVEL1EXTENSIONS_DOTXAXPYF_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_DOTXAXPYF_TCC 1

#include <ulmblas/level1extensions/dotxaxpyf.h>
#include <ulmblas/level1extensions/kernel/dotxaxpyf.h>

namespace ulmBLAS {

template <typename T>
int
dotxaxpyf_fusefactor()
{
    return SELECT_DOTXAXPYF_KERNEL::dotxaxpyf_fusefactor<T>();
}

template <typename IndexType, typename Alpha, typename VA, typename MX,
          typename VY, typename VZ, typename Rho>
void
dotxaxpyf(IndexType      n,
          bool           conjX,
          bool           conjXt,
          bool           conjY,
          const Alpha    &alpha,
          const VA       *a,
          IndexType      incA,
          const MX       *X,
          IndexType      incRowX,
          IndexType      incColX,
          const VY       *y,
          IndexType      incY,
          VZ             *z,
          IndexType      incZ,
          Rho            *rho,
          IndexType      incRho)
{
    SELECT_DOTXAXPYF_KERNEL::dotxaxpyf(n, conjX, conjXt, conjY, alpha, a, incA,
                                       X, incRowX, incColX, y, incY,
                                       z, incZ, rho, incRho);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_DOTXAXPYF_TCC 1
