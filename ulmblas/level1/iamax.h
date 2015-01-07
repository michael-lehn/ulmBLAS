#ifndef ULMBLAS_LEVEL1_IAMAX_H
#define ULMBLAS_LEVEL1_IAMAX_H 1

#include <complex>

namespace ulmBLAS {

template <typename IndexType, typename VX>
    IndexType
    iamax(IndexType      n,
          const VX       *x,
          IndexType      incX);

template <typename IndexType, typename VX>
    IndexType
    iamax(IndexType               n,
          const std::complex<VX>  *x,
          IndexType               incX);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_IAMAX_H 1

#include <ulmblas/level1/iamax.tcc>
