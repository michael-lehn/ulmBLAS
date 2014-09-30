#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_AXPYF_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_AXPYF_TCC 1

#include <ulmblas/level1extensions/kernel/ref/axpyf.h>

#include <iostream>

namespace ulmBLAS { namespace ref {

template <typename T>
int
axpyf_fusefactor()
{
    if (std::is_same<T,double>::value) {
        return 4;
    }
    return 1;
}

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
      IndexType      incY)
{
    const VX *x0 = &x[0*incColX];
    const VX *x1 = &x[1*incColX];
    const VX *x2 = &x[2*incColX];
    const VX *x3 = &x[3*incColX];

    const VA &a0 = a[0*incA];
    const VA &a1 = a[1*incA];
    const VA &a2 = a[2*incA];
    const VA &a3 = a[3*incA];

    for (IndexType i=0; i<n; ++i) {
        y[i*incY] += alpha*(a0*x0[i*incRowX]+a1*x1[i*incRowX]
                           +a2*x2[i*incRowX]+a3*x3[i*incRowX]);
    }
}

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_AXPYF_TCC
