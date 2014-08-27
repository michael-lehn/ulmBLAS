#ifndef SRC_AUXILIARY_PRINTMATRIX_TCC
#define SRC_AUXILIARY_PRINTMATRIX_TCC 1

#include <cstdio>
#include <src/auxiliary/printmatrix.h>

namespace ulmBLAS {

template <typename T, typename IndexType>
void
printMatrix(IndexType m, IndexType n,
            const T *X, IndexType incRowX, IndexType incColX)
{
    for (IndexType i=0; i<m; ++i) {
        for (IndexType j=0; j<n; ++j) {
            printf("  %20.16lf", X[i*incRowX+j*incColX]);
        }
        printf("\n");
    }
    printf("\n");
}

} // namespace ulmBLAS

#endif // SRC_AUXILIARY_PRINTMATRIX_TCC
