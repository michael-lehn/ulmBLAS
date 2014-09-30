#ifndef ULMBLAS_LEVEL1EXTENSIONS_TRLAXPY_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_TRLAXPY_TCC 1

#include <algorithm>
#include <ulmblas/level1/axpy.h>
#include <ulmblas/level1extensions/trlaxpy.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename MX, typename MY>
void
trlaxpy(IndexType    m,
        IndexType    n,
        bool         unit,
        const Alpha  &alpha,
        MX           *X,
        IndexType    incRowX,
        IndexType    incColX,
        MY           *Y,
        IndexType    incRowY,
        IndexType    incColY)
{
    const IndexType    UnitStride(1);

    if (m<=0 || n<=0 || alpha==Alpha(0)) {
        return;
    }

    if (unit) {
        trlaxpy(m-1, n-1, false, alpha,
                &X[1*incRowX], incRowX, incColX,
                &Y[1*incRowY], incRowY, incColY);
        return;
    }

    if (incRowX==UnitStride && incRowY==UnitStride) {
        const IndexType k = std::min(m, n);
        for (IndexType j=0; j<k; ++j) {
            axpy(m-j, alpha,
                 &X[j*(incRowX+incColX)], UnitStride,
                 &Y[j*(incRowY+incColY)], UnitStride);
        }
    } else if (incColX==UnitStride && incColY==UnitStride) {
        for (IndexType i=0; i<m; ++i) {
            axpy(std::min(i+1,n), alpha,
                 &X[i*incRowX], UnitStride,
                 &Y[i*incRowY], UnitStride);
        }
    } else {
        const IndexType k = std::min(m, n);
        for (IndexType j=0; j<k; ++j) {
            for (IndexType i=j; i<m; ++i) {
                Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_TRLAXPY_TCC 1
