#ifndef ULMBLAS_LEVEL1_KERNEL_REF_AXPY_H
#define ULMBLAS_LEVEL1_KERNEL_REF_AXPY_H 1

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename Alpha, typename VX, typename VY>
    void
    axpy(IndexType      n,
         const Alpha    &alpha,
         const VX       *x,
         IndexType      incX,
         VY             *y,
         IndexType      incY);

template <typename IndexType, typename Alpha, typename VX, typename VY>
    void
    acxpy(IndexType      n,
          const Alpha    &alpha,
          const VX       *x,
          IndexType      incX,
          VY             *y,
          IndexType      incY);


} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1_KERNEL_REF_AXPY_H 1

#include <ulmblas/level1/kernel/ref/axpy.tcc>
