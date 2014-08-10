#ifndef ULMBLAS_SRC_LEVEL1_ASUM_H
#define ULMBLAS_SRC_LEVEL1_ASUM_H 1

#include <complex>

namespace ulmBLAS {

template <typename IndexType, typename VX>
    VX
    asum(IndexType  n,
         const VX   *x,
         IndexType  incX);

template <typename IndexType, typename VX>
    VX
    asum(IndexType                n,
         const std::complex<VX>   *x,
         IndexType                incX);

template <typename IndexType, typename VX, typename Result>
    void
    asum(IndexType  n,
         const VX   *x,
         IndexType  incX,
         Result     &result);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_ASUM_H 1
