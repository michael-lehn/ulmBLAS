#ifndef ULMBLAS_LEVEL2_TRLSV_TCC
#define ULMBLAS_LEVEL2_TRLSV_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1extensions/axpyf.h>
#include <ulmblas/level1extensions/dotxf.h>
#include <ulmblas/level2/gemv.h>
#include <ulmblas/level2/trlmv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
trlsv_unblk(IndexType    n,
            bool         unitDiag,
            bool         conjA,
            const TA     *A,
            IndexType    incRowA,
            IndexType    incColA,
            TX           *x,
            IndexType    incX)
{
    for (IndexType i=0; i<n; ++i) {
        for (IndexType j=0; j<i; ++j) {
            x[i*incX] -= conjugate(A[i*incRowA+j*incColA], conjA)*x[j*incX];
        }
        x[i*incX] = (!unitDiag)
                  ? x[i*incX] / conjugate(A[i*incRowA+i*incColA], conjA)
                  : x[i*incX];
    }
}

template <typename IndexType, typename TA, typename TX>
void
trlsv(IndexType    n,
      bool         unitDiag,
      bool         conjA,
      const TA     *A,
      IndexType    incRowA,
      IndexType    incColA,
      TX           *x,
      IndexType    incX)
{
    typedef decltype(TA(0)*TX(0))  T;

    const IndexType    UnitStride(1);

    if (incRowA==UnitStride) {
        const IndexType bf = FuseFactor<T>::axpyf;
        const IndexType nb = (n/bf)*bf;
        const IndexType nl = n % bf;

        for (IndexType j=0; j<nb; j+=bf) {
            trlsv_unblk(bf, unitDiag, conjA,
                        &A[j*UnitStride+j*incColA], UnitStride, incColA,
                        &x[j*incX], incX);

            gemv(n-j-bf, bf,
                 T(-1), conjA,
                 &A[(j+bf)*UnitStride+j*incColA], UnitStride, incColA,
                 &x[j*incX], incX,
                 T(1),
                 &x[(j+bf)*incX], incX);
        }

        trlsv_unblk(nl, unitDiag, conjA,
                    &A[nb*UnitStride+nb*incColA], UnitStride, incColA,
                    &x[nb*incX], incX);

    } else if (incColA==UnitStride) {
        const IndexType bf = FuseFactor<T>::dotuxf;
        const IndexType nb = (n/bf)*bf;
        const IndexType nl = n % bf;

        for (IndexType j=0; j<nb; j+=bf) {
            gemv(bf, j,
                 T(-1), conjA,
                 &A[j*incRowA+0*UnitStride], incRowA, UnitStride,
                 &x[0*incX], incX,
                 T(1),
                 &x[j*incX], incX);

            trlsv_unblk(bf, unitDiag, conjA,
                        &A[j*incRowA+j*UnitStride], incRowA, UnitStride,
                        &x[j*incX], incX);
        }

        if (nl) {
            gemv(nl, n-nl,
                 T(-1), conjA,
                 &A[(n-nl)*incRowA+0*UnitStride], incRowA, UnitStride,
                 &x[0*incX], incX,
                 T(1),
                 &x[(n-nl)*incX], incX);

            trlsv_unblk(nl, unitDiag, conjA,
                        &A[(n-nl)*(incRowA+UnitStride)], incRowA, UnitStride,
                        &x[(n-nl)*incX], incX);
        }

    } else {
        // TODO: Consider blocking
        trlsv_unblk(n, unitDiag, conjA, A, incRowA, incColA, x, incX);
    }
}

template <typename IndexType, typename TA, typename TX>
void
trlsv(IndexType    n,
      bool         unitDiag,
      const TA     *A,
      IndexType    incRowA,
      IndexType    incColA,
      TX           *x,
      IndexType    incX)
{
    trlsv(n, unitDiag, false, A, incRowA, incColA, x, incX);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TRLSV_TCC
