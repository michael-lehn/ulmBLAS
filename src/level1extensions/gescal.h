#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_GESCAL_H
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_GESCAL_H 1

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

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_GESCAL_H 1
