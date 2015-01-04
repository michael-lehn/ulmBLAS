#ifndef ULMBLAS_LEVEL2_SYLR_TCC
#define ULMBLAS_LEVEL2_SYLR_TCC 1

#include <ulmblas/level2/sylr.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TA>
void
sylr(IndexType    n,
     const Alpha  &alpha,
     const TX     *x,
     IndexType    incX,
     TA           *A,
     IndexType    incRowA,
     IndexType    incColA)
{
    /*
    const IndexType    UnitStride(1);

    if (incRowA==UnitStride) {
        return;
    }

    if (incColA==UnitStride) {
        return;
    }
    */

    for (IndexType j=0; j<n; ++j) {
        for (IndexType i=j; i<n; ++i) {
            A[i*incRowA+j*incColA] += alpha*x[i*incX]*x[j*incX];
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_SYLR_TCC
