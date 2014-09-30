#ifndef ULMBLAS_LEVEL1_SCAL_H
#define ULMBLAS_LEVEL1_SCAL_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename VX>
    void
    scal(IndexType      n,
         const Alpha    &alpha,
         VX             *x,
         IndexType      incX);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_SCAL_H 1

#include <ulmblas/level1/scal.tcc>
