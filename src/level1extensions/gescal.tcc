#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_GESCAL_TCC
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_GESCAL_TCC 1

#include <complex>
#include <src/level1extensions/gescal.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename MA>
void
gescal(IndexType    m,
       IndexType    n,
       const Alpha  &alpha,
       MA           *A,
       IndexType    incRowA,
       IndexType    incColA)
{
    for (IndexType j=0; j<n; ++j) {
        for (IndexType i=0; i<m; ++i) {
            A[i*incRowA+j*incColA] *= alpha;
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_GESCAL_TCC 1
