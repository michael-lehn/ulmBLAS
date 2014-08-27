#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_DOTXF_H
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_DOTXF_H 1

namespace ulmBLAS {

template <typename T>
    int
    dotuxf_fusefactor();

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

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_DOTXF_H 1

#include <src/level1extensions/dotxf.tcc>
