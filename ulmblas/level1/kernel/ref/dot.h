#ifndef ULMBLAS_LEVEL1_KERNEL_REF_DOT_H
#define ULMBLAS_LEVEL1_KERNEL_REF_DOT_H 1

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename VX, typename VY, typename Result>
    void
    dotu(IndexType      n,
         const VX       *x,
         IndexType      incX,
         const VY       *y,
         IndexType      incY,
         Result         &result);

template <typename IndexType, typename VT>
    VT
    dotc(IndexType      n,
         const VT       *x,
         IndexType      incX,
         const VT       *y,
         IndexType      incY);

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1_KERNEL_REF_DOT_H 1

#include <ulmblas/level1/kernel/ref/dot.tcc>
