#ifndef ULMBLAS_LEVEL2_TRLMV_TCC
#define ULMBLAS_LEVEL2_TRLMV_TCC 1

#include <ulmblas/level1extensions/axpyf.h>
#include <ulmblas/level1extensions/dotxf.h>
#include <ulmblas/level2/gemv.h>
#include <ulmblas/level2/trlmv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
trlmv_unblk(IndexType    n,
            bool         unitDiag,
            const TA     *A,
            IndexType    incRowA,
            IndexType    incColA,
            TX           *x,
            IndexType    incX)
{
    for (IndexType i=n-1; i>=0; --i) {
        x[i*incX] = (!unitDiag) ? A[i*incRowA+i*incColA]*x[i*incX]
                                : x[i*incX];
        for (IndexType j=0; j<i; ++j) {
            x[i*incX] += A[i*incRowA+j*incColA]*x[j*incX];
        }
    }
}

template <typename IndexType, typename TA, typename TX>
void
trlmv(IndexType    n,
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
        const IndexType nl = n % bf;

        for (IndexType j=n-bf; j>=0; j-=bf) {
            gemv(n-j-bf, bf,
                 T(1),
                 &A[(j+bf)*UnitStride+j*incColA], UnitStride, incColA,
                 &x[ j    *incX], incX,
                 T(1),
                 &x[(j+bf)*incX], incX);

            trlmv_unblk(bf, unitDiag,
                        &A[j*UnitStride+j*incColA], UnitStride, incColA,
                        &x[j*incX], incX);
        }
        if (nl) {
            gemv(n-nl, nl,
                 T(1),
                 &A[nl*UnitStride+0*incColA], UnitStride, incColA,
                 &x[0 *incX], incX,
                 T(1),
                 &x[nl*incX], incX);

            trlmv_unblk(nl, unitDiag,
                        &A[0*UnitStride+0*incColA], UnitStride, incColA,
                        &x[0*incX], incX);
        }
    } else if (incColA==UnitStride) {
        const IndexType bf = dotuxf_fusefactor<T>();
        const IndexType nl = n % bf;

        for (IndexType j=n-bf; j>=0; j-=bf) {
            trlmv_unblk(bf, unitDiag,
                        &A[j*incRowA+j*UnitStride], incRowA, UnitStride,
                        &x[j*incX], incX);

            gemv(bf, j,
                 T(1),
                 &A[j*incRowA], incRowA, UnitStride,
                 &x[0*incX], incX,
                 T(1),
                 &x[j*incX], incX);
        }
        trlmv_unblk(nl, unitDiag,
                    &A[0*incRowA+0*UnitStride], incRowA, UnitStride,
                    &x[0*incX], incX);
    } else {
        // TODO: Consider packing
        trlmv_unblk(n, unitDiag, A, incRowA, incColA, x, incX);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TRLMV_TCC
