#ifndef ULMBLAS_SRC_LEVEL1_REF_AXPY_H
#define ULMBLAS_SRC_LEVEL1_REF_AXPY_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename VX, typename VY>
    void
    axpy_ref(IndexType      n,
             const Alpha    &alpha,
             const VX       *x,
             IndexType      incX,
             VY             *y,
             IndexType      incY);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_REF_AXPY_H 1

#include <src/level1/ref/axpy.tcc>
