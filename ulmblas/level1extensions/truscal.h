#ifndef ULMBLAS_LEVEL1EXTENSIONS_TRUSCAL_H
#define ULMBLAS_LEVEL1EXTENSIONS_TRUSCAL_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename MA>
void
truscal(IndexType    m,
        IndexType    n,
        bool         unit,
        const Alpha  &alpha,
        MA           *A,
        IndexType    incRowA,
        IndexType    incColA);

} // namespace ulmBLAS

#include <ulmblas/level1extensions/truscal.tcc>

#endif // ULMBLAS_LEVEL1EXTENSIONS_TRUSCAL_H 1

