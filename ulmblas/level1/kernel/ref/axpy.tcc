#ifndef ULMBLAS_LEVEL1_KERNEL_REF_AXPY_TCC
#define ULMBLAS_LEVEL1_KERNEL_REF_AXPY_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1/kernel/ref/axpy.h>

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename Alpha, typename VX, typename VY>
void
axpy(IndexType      n,
     const Alpha    &alpha,
     const VX       *x,
     IndexType      incX,
     VY             *y,
     IndexType      incY)
{
    if (n<=0 || alpha==Alpha(0)) {
        return;
    }

    for (IndexType i=0; i<n; ++i) {
        y[i*incY] += alpha*x[i*incX];
    }
}

template <typename IndexType, typename Alpha, typename VX, typename VY>
void
acxpy(IndexType      n,
      const Alpha    &alpha,
      const VX       *x,
      IndexType      incX,
      VY             *y,
      IndexType      incY)
{
    if (n<=0 || alpha==Alpha(0)) {
        return;
    }

    for (IndexType i=0; i<n; ++i) {
        y[i*incY] += alpha*conjugate(x[i*incX]);
    }
}

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1_KERNEL_REF_AXPY_TCC 1
