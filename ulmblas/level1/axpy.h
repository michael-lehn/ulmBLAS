#ifndef ULMBLAS_LEVEL1_AXPY_H
#define ULMBLAS_LEVEL1_AXPY_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename VX, typename VY>
    void
    axpy(IndexType      n,
         const Alpha    &alpha,
         const VX       *x,
         IndexType      incX,
         VY             *y,
         IndexType      incY);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_AXPY_H 1

#include <ulmblas/level1/axpy.tcc>
