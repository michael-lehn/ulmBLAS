#ifndef ULMBLAS_LEVEL2_TRLSV_TCC
#define ULMBLAS_LEVEL2_TRLSV_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/config/level2.h>
#include <ulmblas/level1/dot.h>
#include <ulmblas/level2/gemv.h>
#include <ulmblas/level2/trlmv.h>

namespace ulmBLAS {

//
//  x = A^{-1} x, A lower triangular: element by element (any strides)
//
template <typename IndexType, typename TA, typename TX>
void
trlsv_unblk(IndexType n, bool unitDiag, bool conjA, const TA *A,
            IndexType incRowA, IndexType incColA, TX *x, IndexType incX)
{
    for (IndexType i = 0; i < n; ++i) {
	for (IndexType j = 0; j < i; ++j) {
	    x[i * incX] -=
	        conjugate(A[i * incRowA + j * incColA], conjA) * x[j * incX];
	}
	x[i * incX] =
	    (!unitDiag)
	        ? x[i * incX] / conjugate(A[i * incRowA + i * incColA], conjA)
	        : x[i * incX];
    }
}

//
//  A column major: blocks of FuseN columns (ulmblas/config/level2.h), the
//  triangle of a block element by element, the rows below it with one pass
//  of the fused gemv kernel (axpy column by column was half as fast for
//  complex numbers: ztrsv n = 100 on heim 8 against 18 GFLOP/s of OpenBLAS)
//
template <typename IndexType, typename TA, typename TX>
void
trlsv_colMajor(IndexType n, bool unitDiag, bool conjA, const TA *A,
               IndexType incColA, TX *x, IndexType incX)
{
    typedef decltype(TA(0) * TX(0)) T;

    constexpr IndexType UnitStride(1);
    const IndexType bf = IndexType(Level2Param<T>::FuseN);

    for (IndexType j = 0; j < n; j += bf) {
	const IndexType nb = n - j < bf ? n - j : bf;

	trlsv_unblk(nb, unitDiag, conjA, &A[j + j * incColA], UnitStride,
	            incColA, &x[j * incX], incX);
	gemvSerial(n - j - nb, nb, T(-1), conjA, &A[(j + nb) + j * incColA],
	           UnitStride, incColA, &x[j * incX], incX, T(1),
	           &x[(j + nb) * incX], incX);
    }
}

//
//  A row major: row by row, the solved part with dot
//
template <typename IndexType, typename TA, typename TX>
void
trlsv_rowMajor(IndexType n, bool unitDiag, bool conjA, const TA *A,
               IndexType incRowA, TX *x, IndexType incX)
{
    typedef decltype(TA(0) * TX(0)) T;

    for (IndexType i = 0; i < n; ++i) {
	if (i > 0) {
	    T d;
	    if (conjA) {
		dotc(i, &A[i * incRowA], IndexType(1), x, incX, d);
	    } else {
		dotu(i, &A[i * incRowA], IndexType(1), x, incX, d);
	    }
	    x[i * incX] -= d;
	}
	if (!unitDiag) {
	    x[i * incX] /= conjugate(A[i * incRowA + i], conjA);
	}
    }
}

//
//  Blocks of TrsvBlock columns (ulmblas/config/level2.h; OpenBLAS: 64):
//  the diagonal block as above, the rest of its column panel (A column
//  major) or row panel (row major) with one gemv on one thread.
//
template <typename IndexType, typename TA, typename TX>
void
trlsv(IndexType n, bool unitDiag, bool conjA, const TA *A, IndexType incRowA,
      IndexType incColA, TX *x, IndexType incX)
{
    typedef decltype(TA(0) * TX(0)) T;

    constexpr IndexType UnitStride(1);
    const IndexType bs = IndexType(Level2Param<T>::TrsvBlock);

    if (incRowA == UnitStride) {
	for (IndexType j = 0; j < n; j += bs) {
	    const IndexType nb = n - j < bs ? n - j : bs;

	    trlsv_colMajor(nb, unitDiag, conjA, &A[j + j * incColA], incColA,
	                   &x[j * incX], incX);
	    gemvSerial(n - j - nb, nb, T(-1), conjA, &A[(j + nb) + j * incColA],
	               UnitStride, incColA, &x[j * incX], incX, T(1),
	               &x[(j + nb) * incX], incX);
	}
    } else if (incColA == UnitStride) {
	for (IndexType j = 0; j < n; j += bs) {
	    const IndexType nb = n - j < bs ? n - j : bs;

	    gemvSerial(nb, j, T(-1), conjA, &A[j * incRowA], incRowA,
	               UnitStride, x, incX, T(1), &x[j * incX], incX);
	    trlsv_rowMajor(nb, unitDiag, conjA, &A[j * incRowA + j], incRowA,
	                   &x[j * incX], incX);
	}
    } else {
	trlsv_unblk(n, unitDiag, conjA, A, incRowA, incColA, x, incX);
    }
}

template <typename IndexType, typename TA, typename TX>
void
trlsv(IndexType n, bool unitDiag, const TA *A, IndexType incRowA,
      IndexType incColA, TX *x, IndexType incX)
{
    trlsv(n, unitDiag, false, A, incRowA, incColA, x, incX);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TRLSV_TCC
