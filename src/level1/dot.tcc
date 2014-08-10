#ifndef ULMBLAS_SRC_LEVEL1_DOT_TCC
#define ULMBLAS_SRC_LEVEL1_DOT_TCC 1

#include <complex>
#include <src/level1/dot.h>

namespace ulmBLAS {

template <typename IndexType, typename VT>
VT
dotu(IndexType      n,
     const VT       *x,
     IndexType      incX,
     const VT       *y,
     IndexType      incY)
{
    VT result;

    dotu(n, x, incX, y, incY, result);
    return result;
}

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

template <typename IndexType, typename VT>
VT
dotc(IndexType      n,
     const VT       *x,
     IndexType      incX,
     const VT       *y,
     IndexType      incY)
{
    VT result;

    dotc(n, x, incX, y, incY, result);
    return result;
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

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_DOT_TCC 1
