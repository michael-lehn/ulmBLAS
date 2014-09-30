#ifndef ULMBLAS_LEVEL1_SWAP_H
#define ULMBLAS_LEVEL1_SWAP_H 1

namespace ulmBLAS {

template <typename IndexType, typename VX, typename VY>
    void
    swap(IndexType      n,
         VX             *x,
         IndexType      incX,
         VY             *y,
         IndexType      incY);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_SWAP_H 1

#include <ulmblas/level1/swap.tcc>
