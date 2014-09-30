#ifndef ULMBLAS_LEVEL1EXTENSIONS_TRUSCAL_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_TRUSCAL_TCC 1

#include <algorithm>
#include <ulmblas/level1/scal.h>
#include <ulmblas/level1extensions/truscal.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename MA>
void
truscal(IndexType    m,
        IndexType    n,
        bool         unit,
        const Alpha  &alpha,
        MA           *A,
        IndexType    incRowA,
        IndexType    incColA)
{
    const IndexType    UnitStride(1);

    if (alpha==Alpha(1) || m<=0 || n<=0) {
        return;
    }

    if (unit) {
        truscal(m, n-1, false, alpha, &A[1*incColA], incRowA, incColA);
        return;
    }

    if (incRowA==UnitStride) {
        for (IndexType j=0; j<n; ++j) {
            scal(std::min(j+1,m), alpha, &A[j*incColA], UnitStride);
        }
    } else if (incColA==UnitStride) {
        const IndexType k = std::min(m, n);
        for (IndexType i=0; i<k; ++i) {
            scal(n-i, alpha, &A[i*(incRowA+incColA)], UnitStride);
        }
    } else {
        const IndexType k = std::min(m, n);
        for (IndexType i=0; i<k; ++i) {
            for (IndexType j=i; j<n; ++j) {
                A[i*incRowA+j*incColA] *= alpha;
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_TRUSCAL_TCC 1

