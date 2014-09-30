#ifndef ULMBLAS_LEVEL1EXTENSIONS_TRUAXPY_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_TRUAXPY_TCC 1

#include <algorithm>
#include <ulmblas/level1/axpy.h>
#include <ulmblas/level1extensions/trlaxpy.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename MX, typename MY>
void
truaxpy(IndexType    m,
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
        truaxpy(m-1, n-1, false, alpha,
                &X[1*incColX], incRowX, incColX,
                &Y[1*incColY], incRowY, incColY);
        return;
    }

    if (incRowX==UnitStride && incRowY==UnitStride) {
        for (IndexType j=0; j<n; ++j) {
            axpy(std::min(j+1,m), alpha,
                 &X[j*incColX], UnitStride,
                 &Y[j*incColY], UnitStride);
        }
    } else if (incColX==UnitStride && incColY==UnitStride) {
        const IndexType k = std::min(m, n);
        for (IndexType i=0; i<k; ++i) {
            axpy(n-i, alpha,
                 &X[i*(incRowX+incColX)], UnitStride,
                 &Y[i*(incRowY+incColY)], UnitStride);
        }
    } else {
        for (IndexType j=0; j<n; ++j) {
            for (IndexType i=0; i<std::min(j+1,m); ++i) {
                Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_TRUAXPY_TCC 1
