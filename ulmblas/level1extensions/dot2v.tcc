#ifndef ULMBLAS_LEVEL1EXTENSIONS_DOT2V_H
#define ULMBLAS_LEVEL1EXTENSIONS_DOT2V_H 1

#include <ulmblas/level1extensions/dot2v.h>
#include <ulmblas/level1extensions/kernel/dot2v.h>

namespace ulmBLAS {

template <typename IndexType, typename VX0, typename VX1, typename VY,
          typename Result>
void
dotu2v(IndexType      n,
       const VX0      *x0,
       IndexType      incX0,
       const VX1      *x1,
       IndexType      incX1,
       VY             *y,
       IndexType      incY,
       Result         *result,
       IndexType      resultInc)
{
    SELECT_DOT2V_KERNEL::dotu2v(n, x0, incX0, x1, incX1, y, incY,
                                result, resultInc);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_DOT2V_H
