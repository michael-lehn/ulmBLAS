#ifndef ULMBLAS_LEVEL1EXTENSIONS_GESCAL_H
#define ULMBLAS_LEVEL1EXTENSIONS_GESCAL_H 1

#include <complex>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename MA>
    void
    gescal(IndexType    m,
           IndexType    n,
           const Alpha  &alpha,
           MA           *A,
           IndexType    incRowA,
           IndexType    incColA);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_GESCAL_H 1

#include <ulmblas/level1extensions/gescal.tcc>
