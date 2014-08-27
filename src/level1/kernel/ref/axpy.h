#ifndef ULMBLAS_SRC_LEVEL1_KERNEL_REF_AXPY_H
#define ULMBLAS_SRC_LEVEL1_KERNEL_REF_AXPY_H 1

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename Alpha, typename VX, typename VY>
    void
    axpy(IndexType      n,
         const Alpha    &alpha,
         const VX       *x,
         IndexType      incX,
         VY             *y,
         IndexType      incY);

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_KERNEL_REF_AXPY_H 1

#include <src/level1/kernel/ref/axpy.tcc>
