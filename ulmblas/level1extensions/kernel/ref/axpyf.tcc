#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_AXPYF_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_AXPYF_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1extensions/kernel/ref/axpyf.h>

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename Alpha, typename VA, typename VX,
           typename VY>
typename std::enable_if<std::is_integral<IndexType>::value
                  && FuseFactor<decltype(Alpha(0)*VA(0)*VX(0)+VY(0))>::axpyf==4,
    void>::type
axpyf(IndexType      n,
      const Alpha    &alpha,
      const VA       *a,
      IndexType      incA,
      const VX       *X,
      IndexType      incRowX,
      IndexType      incColX,
      VY             *y,
      IndexType      incY)
{
    const VX *x0 = &X[0*incColX];
    const VX *x1 = &X[1*incColX];
    const VX *x2 = &X[2*incColX];
    const VX *x3 = &X[3*incColX];

    const VA &a0 = a[0*incA];
    const VA &a1 = a[1*incA];
    const VA &a2 = a[2*incA];
    const VA &a3 = a[3*incA];

    for (IndexType i=0; i<n; ++i) {
        y[i*incY] += alpha*(a0*x0[i*incRowX]+a1*x1[i*incRowX]
                           +a2*x2[i*incRowX]+a3*x3[i*incRowX]);
    }
}

template <typename IndexType, typename Alpha, typename VA, typename VX,
           typename VY>
typename std::enable_if<std::is_integral<IndexType>::value
                  && FuseFactor<decltype(Alpha(0)*VA(0)*VX(0)+VY(0))>::axpyf!=4,
    void>::type
axpyf(IndexType      n,
      const Alpha    &alpha,
      const VA       *a,
      IndexType      incA,
      const VX       *X,
      IndexType      incRowX,
      IndexType      incColX,
      VY             *y,
      IndexType      incY)
{
    typedef decltype(Alpha(0)*VA(0)*VX(0)+VY(0))    T;

    const IndexType ff = FuseFactor<T>::axpyf;

    for (IndexType i=0; i<n; ++i) {
        for (IndexType j=0; j<ff; ++j) {
            y[i*incY] += alpha*a[j*incA]*X[i*incRowX+j*incColX];
        }
    }
}


template <typename IndexType, typename Alpha, typename VA, typename VX,
           typename VY>
typename std::enable_if<std::is_integral<IndexType>::value
                 && FuseFactor<decltype(Alpha(0)*VA(0)*VX(0)+VY(0))>::acxpyf==4,
    void>::type
acxpyf(IndexType      n,
       const Alpha    &alpha,
       const VA       *a,
       IndexType      incA,
       const VX       *X,
       IndexType      incRowX,
       IndexType      incColX,
       VY             *y,
       IndexType      incY)
{
    const VX *x0 = &X[0*incColX];
    const VX *x1 = &X[1*incColX];
    const VX *x2 = &X[2*incColX];
    const VX *x3 = &X[3*incColX];

    const VA &a0 = a[0*incA];
    const VA &a1 = a[1*incA];
    const VA &a2 = a[2*incA];
    const VA &a3 = a[3*incA];

    for (IndexType i=0; i<n; ++i) {
        y[i*incY] += alpha*(a0*conjugate(x0[i*incRowX])
                           +a1*conjugate(x1[i*incRowX])
                           +a2*conjugate(x2[i*incRowX])
                           +a3*conjugate(x3[i*incRowX]));
    }
}

template <typename IndexType, typename Alpha, typename VA, typename VX,
           typename VY>
typename std::enable_if<std::is_integral<IndexType>::value
                 && FuseFactor<decltype(Alpha(0)*VA(0)*VX(0)+VY(0))>::acxpyf!=4,
    void>::type
acxpyf(IndexType      n,
       const Alpha    &alpha,
       const VA       *a,
       IndexType      incA,
       const VX       *X,
       IndexType      incRowX,
       IndexType      incColX,
       VY             *y,
       IndexType      incY)
{
    typedef decltype(Alpha(0)*VA(0)*VX(0)+VY(0))    T;

    const IndexType ff = FuseFactor<T>::axpyf;

    for (IndexType i=0; i<n; ++i) {
        for (IndexType j=0; j<ff; ++j) {
            y[i*incY] += alpha*a[j*incA]*conjugate(X[i*incRowX+j*incColX]);
        }
    }
}

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_AXPYF_TCC
