#ifndef ULMBLAS_SRC_LEVEL1_AXPY_TCC
#define ULMBLAS_SRC_LEVEL1_AXPY_TCC 1

#include <src/level1/axpy.h>
#include <src/level1/kernel/axpy.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename VX, typename VY>
void
axpy(IndexType      n,
     const Alpha    &alpha,
     const VX       *x,
     IndexType      incX,
     VY             *y,
     IndexType      incY)
{
    SELECT_AXPY_KERNEL::axpy(n, alpha, x, incX, y, incY);
}

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_AXPY_TCC 1
