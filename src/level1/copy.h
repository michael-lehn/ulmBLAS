#ifndef ULMBLAS_SRC_LEVEL1_COPY_H
#define ULMBLAS_SRC_LEVEL1_COPY_H 1

namespace ulmBLAS {

template <typename IndexType, typename VX, typename VY>
    void
    copy(IndexType      n,
         const VX       *x,
         IndexType      incX,
         VY             *y,
         IndexType      incY);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_COPY_H 1

#include <src/level1/copy.tcc>
