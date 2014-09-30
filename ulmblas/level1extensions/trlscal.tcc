#ifndef ULMBLAS_LEVEL1EXTENSIONS_TRLSCAL_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_TRLSCAL_TCC 1

#include <algorithm>
#include <ulmblas/level1/scal.h>
#include <ulmblas/level1extensions/trlscal.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename MA>
void
trlscal(IndexType    m,
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
        trlscal(m-1, n, false, alpha, &A[1*incRowA], incRowA, incColA);
        return;
    }

    if (incRowA==UnitStride) {
        const IndexType k = std::min(m, n);
        for (IndexType j=0; j<k; ++j) {
            scal(m-j, alpha, &A[j*(incRowA+incColA)], UnitStride);
        }
    } else if (incColA==UnitStride) {
        for (IndexType i=0; i<m; ++i) {
            scal(std::min(i+1,n), alpha, &A[i*incRowA], UnitStride);
        }
    } else {
        const IndexType k = std::min(m, n);
        for (IndexType j=0; j<k; ++j) {
            for (IndexType i=j; i<m; ++i) {
                A[i*incRowA+j*incColA] *= alpha;
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_TRLSCAL_TCC 1

