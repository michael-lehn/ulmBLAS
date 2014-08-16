#ifndef ULMBLAS_SRC_LEVEL1_SCAL_H
#define ULMBLAS_SRC_LEVEL1_SCAL_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename VX>
    void
    scal(IndexType      n,
         const Alpha    &alpha,
         VX             *x,
         IndexType      incX);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_SCAL_H 1

#include <src/level1/scal.tcc>
