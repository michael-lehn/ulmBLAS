#ifndef ULMBLAS_SRC_LEVEL1_REF_DOT_H
#define ULMBLAS_SRC_LEVEL1_REF_DOT_H 1

namespace ulmBLAS {

template <typename IndexType, typename VX, typename VY, typename Result>
    void
    dotu_ref(IndexType      n,
             const VX       *x,
             IndexType      incX,
             const VY       *y,
             IndexType      incY,
             Result         &result);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_REF_DOT_H 1

#include <src/level1/ref/dot.tcc>
