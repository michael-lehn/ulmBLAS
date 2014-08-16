#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_REF_AXPY2V_H
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_REF_AXPY2V_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha0, typename Alpha1,
          typename VX0, typename VX1, typename VY>
    void
    axpy2v_ref(IndexType      n,
               const Alpha0   &alpha0,
               const Alpha1   &alpha1,
               const VX0      *x0,
               IndexType      incX0,
               const VX1      *x1,
               IndexType      incX1,
               VY             *y,
               IndexType      incY);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_REF_AXPY2V_H 1

#include <src/level1extensions/ref/axpy2v.tcc>
