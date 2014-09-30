#ifndef ULMBLAS_LEVEL2_TRUMV_TCC
#define ULMBLAS_LEVEL2_TRUMV_TCC 1

#include <ulmblas/level1extensions/axpyf.h>
#include <ulmblas/level1extensions/dotxf.h>
#include <ulmblas/level2/gemv.h>
#include <ulmblas/level2/trumv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
trumv_unblk(IndexType    n,
            bool         unitDiag,
            const TA     *A,
            IndexType    incRowA,
            IndexType    incColA,
            TX           *x,
            IndexType    incX)
{
    for (IndexType i=0; i<n; ++i) {
        x[i*incX] = (!unitDiag) ? A[i*incRowA+i*incColA]*x[i*incX]
                                : x[i*incX];
        for (IndexType j=i+1; j<n; ++j) {
            x[i*incX] += A[i*incRowA+j*incColA]*x[j*incX];
        }
    }
}

template <typename IndexType, typename TA, typename TX>
void
trumv(IndexType    n,
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
            gemv(j, bf,
                 T(1),
                 &A[0*UnitStride+j*incColA], UnitStride, incColA,
                 &x[j*incX], incX,
                 T(1),
                 &x[0*incX], incX);

            trumv_unblk(bf, unitDiag,
                        &A[j*UnitStride+j*incColA], UnitStride, incColA,
                        &x[j*incX], incX);
        }

        if (nl) {
            gemv(n-nl, nl,
                 T(1),
                 &A[0*UnitStride+(n-nl)*incColA], UnitStride, incColA,
                 &x[(n-nl)*incX], incX,
                 T(1),
                 &x[0*incX], incX);

            trlmv_unblk(nl, unitDiag,
                      &A[(n-nl)*UnitStride+(n-nl)*incColA], UnitStride, incColA,
                      &x[(n-nl)*incX], incX);
        }

    } else if (incColA==UnitStride) {
        const IndexType bf = dotuxf_fusefactor<T>();
        const IndexType nb = (n/bf)*bf;
        const IndexType nl = n % bf;

        for (IndexType j=0; j<nb; j+=bf) {
            trumv_unblk(bf, unitDiag,
                        &A[j*incRowA+j*UnitStride], incRowA, UnitStride,
                        &x[j*incX], incX);

            gemv(bf, n-j-bf,
                 T(1),
                 &A[j*incRowA+(j+bf)*incColA], incRowA, UnitStride,
                 &x[(j+bf)*incX], incX,
                 T(1),
                 &x[j*incX], incX);
        }

        trumv_unblk(nl, unitDiag,
                    &A[(n-nl)*incRowA+(n-nl)*UnitStride], incRowA, UnitStride,
                    &x[(n-nl)*incX], incX);

    } else {
        // TODO: Consider packing
        trumv_unblk(n, unitDiag, A, incRowA, incColA, x, incX);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TRUMV_TCC
