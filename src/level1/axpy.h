#ifndef ULMBLAS_SRC_LEVEL1_AXPY_H
#define ULMBLAS_SRC_LEVEL1_AXPY_H 1

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

#endif // ULMBLAS_SRC_LEVEL1_AXPY_H 1
