#ifndef ULMBLAS_LEVEL1_DOT_TCC
#define ULMBLAS_LEVEL1_DOT_TCC 1

#include <complex>
#include <ulmblas/level1/dot.h>
#include <ulmblas/level1/kernel/dot.h>

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
    SELECT_DOT_KERNEL::dotu(n, x, incX, y, incY, result);
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
    SELECT_DOT_KERNEL::dotc(n, x, incX, y, incY, result);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_DOT_TCC 1
