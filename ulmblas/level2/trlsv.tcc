#ifndef ULMBLAS_LEVEL2_TRLSV_TCC
#define ULMBLAS_LEVEL2_TRLSV_TCC 1

#include <ulmblas/level1extensions/axpyf.h>
#include <ulmblas/level1extensions/dotxf.h>
#include <ulmblas/level2/gemv.h>
#include <ulmblas/level2/trlmv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
trlsv_unblk(IndexType    n,
            bool         unitDiag,
            const TA     *A,
            IndexType    incRowA,
            IndexType    incColA,
            TX           *x,
            IndexType    incX)
{
    for (IndexType i=0; i<n; ++i) {
        for (IndexType j=0; j<i; ++j) {
            x[i*incX] -= A[i*incRowA+j*incColA]*x[j*incX];
        }
        x[i*incX] = (!unitDiag) ? x[i*incX] / A[i*incRowA+i*incColA]
                                : x[i*incX];
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
    typedef decltype(TA(0)*TX(0))  T;

    const IndexType    UnitStride(1);

    if (incRowA==UnitStride) {
        const IndexType bf = axpyf_fusefactor<T>();
        const IndexType nb = (n/bf)*bf;
        const IndexType nl = n % bf;

        for (IndexType j=0; j<nb; j+=bf) {
            trlsv_unblk(bf, unitDiag,
                        &A[j*UnitStride+j*incColA], UnitStride, incColA,
                        &x[j*incX], incX);

            gemv(n-j-bf, bf,
                 T(-1),
                 &A[(j+bf)*UnitStride+j*incColA], UnitStride, incColA,
                 &x[j*incX], incX,
                 T(1),
                 &x[(j+bf)*incX], incX);
        }

        trlsv_unblk(nl, unitDiag,
                    &A[nb*UnitStride+nb*incColA], UnitStride, incColA,
                    &x[nb*incX], incX);

    } else if (incColA==UnitStride) {
        const IndexType bf = dotuxf_fusefactor<T>();
        const IndexType nb = (n/bf)*bf;
        const IndexType nl = n % bf;

        for (IndexType j=0; j<nb; j+=bf) {
            gemv(bf, j,
                 T(-1),
                 &A[j*incRowA+0*UnitStride], incRowA, UnitStride,
                 &x[0*incX], incX,
                 T(1),
                 &x[j*incX], incX);

            trlsv_unblk(bf, unitDiag,
                        &A[j*incRowA+j*UnitStride], incRowA, UnitStride,
                        &x[j*incX], incX);
        }

        if (nl) {
            gemv(nl, n-nl,
                 T(-1),
                 &A[(n-nl)*incRowA+0*UnitStride], incRowA, UnitStride,
                 &x[0*incX], incX,
                 T(1),
                 &x[(n-nl)*incX], incX);

            trlsv_unblk(nl, unitDiag,
                        &A[(n-nl)*(incRowA+UnitStride)], incRowA, UnitStride,
                        &x[(n-nl)*incX], incX);
        }

    } else {
        // TODO: Consider blocking
        trlsv_unblk(n, unitDiag, A, incRowA, incColA, x, incX);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TRLSV_TCC
