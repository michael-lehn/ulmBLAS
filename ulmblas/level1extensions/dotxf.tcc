#ifndef ULMBLAS_LEVEL1EXTENSIONS_DOTXF_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_DOTXF_TCC 1

#include <ulmblas/level1extensions/dotxf.h>
#include <ulmblas/level1extensions/kernel/dotxf.h>

namespace ulmBLAS {

template <typename IndexType, typename TX, typename TY, typename Result>
void
dotuxf(IndexType      n,
       const TX       *X,
       IndexType      incRowX,
       IndexType      incColX,
       const TY       *y,
       IndexType      incY,
       Result         *result,
       IndexType      resultInc)
{
    SELECT_DOTXF_KERNEL::dotuxf(n, X, incRowX, incColX, y, incY,
                                result, resultInc);
}

template <typename IndexType, typename TX, typename TY, typename Result>
void
dotcxf(IndexType      n,
       const TX       *X,
       IndexType      incRowX,
       IndexType      incColX,
       const TY       *y,
       IndexType      incY,
       Result         *result,
       IndexType      resultInc)
{
    SELECT_DOTXF_KERNEL::dotcxf(n, X, incRowX, incColX, y, incY,
                                result, resultInc);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_DOTXF_TCC 1
