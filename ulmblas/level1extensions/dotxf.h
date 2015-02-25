#ifndef ULMBLAS_LEVEL1EXTENSIONS_DOTXF_H
#define ULMBLAS_LEVEL1EXTENSIONS_DOTXF_H 1

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
           IndexType      resultInc);

template <typename IndexType, typename TX, typename TY, typename Result>
    void
    dotcxf(IndexType      n,
           const TX       *X,
           IndexType      incRowX,
           IndexType      incColX,
           const TY       *y,
           IndexType      incY,
           Result         *result,
           IndexType      resultInc);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_DOTXF_H 1

#include <ulmblas/level1extensions/dotxf.tcc>
