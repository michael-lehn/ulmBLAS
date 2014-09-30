#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_AXPYF_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_AXPYF_H 1

namespace ulmBLAS { namespace ref {

template <typename T>
    int
    axpyf_fusefactor();

template <typename IndexType, typename Alpha, typename VA, typename VX,
           typename VY>
    void
    axpyf(IndexType      n,
          const Alpha    &alpha,
          const VA       *a,
          IndexType      incA,
          const VX       *x,
          IndexType      incRowX,
          IndexType      incColX,
          VY             *y,
          IndexType      incY);

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_AXPYF_H

#include <ulmblas/level1extensions/kernel/ref/axpyf.tcc>
