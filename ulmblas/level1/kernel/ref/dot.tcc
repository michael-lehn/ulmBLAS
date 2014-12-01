#ifndef ULMBLAS_LEVEL1_KERNEL_REF_DOT_TCC
#define ULMBLAS_LEVEL1_KERNEL_REF_DOT_TCC 1

#include <complex>
#include <ulmblas/level1/kernel/ref/dot.h>

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename VX, typename VY, typename Result>
void
dotu(IndexType      n,
     const VX       *x,
     IndexType      incX,
     const VY       *y,
     IndexType      incY,
     Result         &result)
{
    result = Result(0);

    for (IndexType i=0; i<n; ++i) {
        result += x[i*incX]*y[i*incY];
    }
}

template <typename IndexType, typename VX, typename VY, typename Result>
void
dotc(IndexType      n,
     const VX       *x,
     IndexType      incX,
     const VY       *y,
     IndexType      incY,
     Result         &result)
{
    result = Result(0);

    for (IndexType i=0; i<n; ++i) {
        result += std::conj(x[i*incX])*y[i*incY];
    }
}

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1_KERNEL_REF_DOT_TCC 1
