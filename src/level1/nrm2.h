#ifndef ULMBLAS_SRC_LEVEL1_NRM2_H
#define ULMBLAS_SRC_LEVEL1_NRM2_H 1

#include <complex>

namespace ulmBLAS {

template <typename IndexType, typename VX>
    VX
    nrm2(IndexType  n,
         const VX   *x,
         IndexType  incX);

template <typename IndexType, typename VX>
    VX
    nrm2(IndexType                n,
         const std::complex<VX>   *x,
         IndexType                incX);

template <typename IndexType, typename VX, typename Result>
    void
    nrm2(IndexType  n,
         const VX   *x,
         IndexType  incX,
         Result     &result);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_NRM2_H 1

#include <src/level1/nrm2.tcc>
