#ifndef ULMBLAS_SRC_LEVEL1_IAMAX_H
#define ULMBLAS_SRC_LEVEL1_IAMAX_H 1

namespace ulmBLAS {

template <typename IndexType, typename VX>
    IndexType
    iamax(IndexType      n,
          const VX       *x,
          IndexType      incX);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_IAMAX_H 1
