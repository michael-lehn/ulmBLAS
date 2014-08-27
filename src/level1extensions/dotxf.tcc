#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_DOTXF_TCC
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_DOTXF_TCC 1

#include <src/level1extensions/dotxf.h>
#include <src/level1extensions/kernel/dotxf.h>

namespace ulmBLAS {

template <typename T>
int
dotuxf_fusefactor()
{
    return SELECT_DOTXF_KERNEL::dotuxf_fusefactor<T>();
}

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

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_DOTXF_TCC 1
