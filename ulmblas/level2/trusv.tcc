#ifndef ULMBLAS_LEVEL2_TRUSV_TCC
#define ULMBLAS_LEVEL2_TRUSV_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/config/level2.h>
#include <ulmblas/level1/dot.h>
#include <ulmblas/level2/gemv.h>
#include <ulmblas/level2/trusv.h>

namespace ulmBLAS {

//
//  x = A^{-1} x, A upper triangular: element by element (any strides)
//
template <typename IndexType, typename TA, typename TX>
void
trusv_unblk(IndexType n, bool unitDiag, bool conjA, const TA *A,
            IndexType incRowA, IndexType incColA, TX *x, IndexType incX)
{
    for (IndexType i = n - 1; i >= 0; --i) {
	for (IndexType j = i + 1; j < n; ++j) {
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
//  A column major: blocks of FuseN columns from the last (see trlsv), the
//  triangle of a block element by element, the rows above it with one pass
//  of the fused gemv kernel
//
template <typename IndexType, typename TA, typename TX>
void
trusv_colMajor(IndexType n, bool unitDiag, bool conjA, const TA *A,
               IndexType incColA, TX *x, IndexType incX)
{
    typedef decltype(TA(0) * TX(0)) T;

    constexpr IndexType UnitStride(1);
    const IndexType bf = IndexType(Level2Param<T>::FuseN);

    for (IndexType j1 = n; j1 > 0; j1 -= bf) {
	const IndexType nb = j1 < bf ? j1 : bf;
	const IndexType j = j1 - nb;

	trusv_unblk(nb, unitDiag, conjA, &A[j + j * incColA], UnitStride,
	            incColA, &x[j * incX], incX);
	gemvSerial(j, nb, T(-1), conjA, &A[j * incColA], UnitStride, incColA,
	           &x[j * incX], incX, T(1), x, incX);
    }
}

//
//  A row major: row by row from the last, the solved part with dot
//
template <typename IndexType, typename TA, typename TX>
void
trusv_rowMajor(IndexType n, bool unitDiag, bool conjA, const TA *A,
               IndexType incRowA, TX *x, IndexType incX)
{
    typedef decltype(TA(0) * TX(0)) T;

    for (IndexType i = n - 1; i >= 0; --i) {
	if (i + 1 < n) {
	    T d;
	    if (conjA) {
		dotc(n - i - 1, &A[i * incRowA + i + 1], IndexType(1),
		     &x[(i + 1) * incX], incX, d);
	    } else {
		dotu(n - i - 1, &A[i * incRowA + i + 1], IndexType(1),
		     &x[(i + 1) * incX], incX, d);
	    }
	    x[i * incX] -= d;
	}
	if (!unitDiag) {
	    x[i * incX] /= conjugate(A[i * incRowA + i], conjA);
	}
    }
}

//
//  Blocks of TrsvBlock columns from the last (ulmblas/config/level2.h):
//  the diagonal block as above, the rest of its column panel (A column
//  major) or row panel (row major) with one gemv on one thread.
//
template <typename IndexType, typename TA, typename TX>
void
trusv(IndexType n, bool unitDiag, bool conjA, const TA *A, IndexType incRowA,
      IndexType incColA, TX *x, IndexType incX)
{
    typedef decltype(TA(0) * TX(0)) T;

    constexpr IndexType UnitStride(1);
    const IndexType bs = IndexType(Level2Param<T>::TrsvBlock);

    if (incRowA == UnitStride) {
	for (IndexType j1 = n; j1 > 0; j1 -= bs) {
	    const IndexType nb = j1 < bs ? j1 : bs;
	    const IndexType j = j1 - nb;

	    trusv_colMajor(nb, unitDiag, conjA, &A[j + j * incColA], incColA,
	                   &x[j * incX], incX);
	    gemvSerial(j, nb, T(-1), conjA, &A[j * incColA], UnitStride,
	               incColA, &x[j * incX], incX, T(1), x, incX);
	}
    } else if (incColA == UnitStride) {
	for (IndexType j1 = n; j1 > 0; j1 -= bs) {
	    const IndexType nb = j1 < bs ? j1 : bs;
	    const IndexType j = j1 - nb;

	    gemvSerial(nb, n - j1, T(-1), conjA, &A[j * incRowA + j1], incRowA,
	               UnitStride, &x[j1 * incX], incX, T(1), &x[j * incX],
	               incX);
	    trusv_rowMajor(nb, unitDiag, conjA, &A[j * incRowA + j], incRowA,
	                   &x[j * incX], incX);
	}
    } else {
	trusv_unblk(n, unitDiag, conjA, A, incRowA, incColA, x, incX);
    }
}

template <typename IndexType, typename TA, typename TX>
void
trusv(IndexType n, bool unitDiag, const TA *A, IndexType incRowA,
      IndexType incColA, TX *x, IndexType incX)
{
    trusv(n, unitDiag, false, A, incRowA, incColA, x, incX);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TRUSV_TCC
