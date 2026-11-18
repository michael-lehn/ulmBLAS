#ifndef ULMBLAS_LAPACK_GETRF2_TCC
#define ULMBLAS_LAPACK_GETRF2_TCC 1

#include <algorithm>
#include <utility>
#include <ulmblas/lapack/islapacktype.h>
#include <ulmblas/lapack/getf2.h>
#include <ulmblas/lapack/getrf2.h>
#include <ulmblas/lapack/unitstride.h>
#include <ulmblas/level3/gemm.h>
#include <ulmblas/level3/trlsm.h>

//  base case sizes (compile time parameters, tuned for double)
#ifndef ULMBLAS_GETRF2_BASE
#define ULMBLAS_GETRF2_BASE 16
#endif
#ifndef ULMBLAS_GETRF2_BASE_SMALL
#define ULMBLAS_GETRF2_BASE_SMALL 32 // for m <= 96
#endif

namespace ulmBLAS {

namespace getrf2_impl {

//
//  Row interchanges (forward): for i = k1, ..., k2-1 rows i and
//  piv[i*incPiv] of the n columns of A are swapped (piv 0-based).  Column
//  by column, which is cache friendly for column-major storage.
//
template <typename IndexType, typename T, typename PivIndex>
void
swapRows(IndexType n, T *A, IndexType rs, IndexType cs, IndexType k1,
         IndexType k2, const PivIndex *piv, IndexType incPiv)
{
    for (IndexType j = 0; j < n; ++j) {
	T *a = &A[j * cs];
	for (IndexType i = k1; i < k2; ++i) {
	    const IndexType p = IndexType(piv[i * incPiv]);
	    if (p != i) {
		std::swap(a[i * rs], a[p * rs]);
	    }
	}
    }
}

//
//  Neither stride 1: the largest blocks of columns that fit into a copy
//  (lapackCopyFits) are factored on a column-major copy, the updates of the
//  larger levels (swaps, trsm, gemm) work on the strides of A.  The pivots
//  are stored in piv directly (no work space).
//
template <typename IndexType, typename T, typename PivIndex>
IndexType
rec(IndexType m, IndexType n, T *A, IndexType rs, IndexType cs, PivIndex *piv,
    IndexType incPiv)
{
    const IndexType kmin = std::min(m, n);
    if (kmin == 0) {
	return 0;
    }
    if (rs != 1 && cs != 1 && lapackCopyFits<T>(m, n)) {
	return withUnitStride(
	    m, n, A, rs, cs, false,
	    [m, n, piv, incPiv](T *B, IndexType brs, IndexType bcs) {
	        return rec(m, n, B, brs, bcs, piv, incPiv);
	    });
    }
    if (n <= (m <= 96 ? ULMBLAS_GETRF2_BASE_SMALL : ULMBLAS_GETRF2_BASE) ||
        m == 1) {
	return getf2(m, n, A, rs, cs, piv, incPiv);
    }
    const IndexType n1 = kmin / 2;
    const IndexType n2 = n - n1;

    IndexType info = rec(m, n1, A, rs, cs, piv, incPiv);

    T *A12 = &A[n1 * cs];
    T *A21 = &A[n1 * rs];
    T *A22 = &A[n1 * (rs + cs)];

    swapRows(n2, A12, rs, cs, IndexType(0), n1, piv, incPiv);
    trlsm(n1, n2, T(1), false, true, A, rs, cs, A12, rs, cs);
    if (m > n1) {
	gemm(m - n1, n2, n1, T(-1), false, A21, rs, cs, false, A12, rs, cs,
	     T(1), A22, rs, cs);
    }

    IndexType info2 = rec(m - n1, n2, A22, rs, cs, &piv[n1 * incPiv], incPiv);
    if (info == 0 && info2 > 0) {
	info = info2 + n1;
    }
    for (IndexType i = n1; i < kmin; ++i) {
	piv[i * incPiv] += PivIndex(n1);
    }
    swapRows(n1, A, rs, cs, n1, kmin, piv, incPiv);
    return info;
}

} // namespace getrf2_impl

template <typename IndexType, typename T, typename PivIndex>
IndexType
getrf2(IndexType m, IndexType n, T *A, IndexType incRowA, IndexType incColA,
       PivIndex *piv, IndexType incPiv)
{
    ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T);

    if (std::min(m, n) <= 0) {
	return 0;
    }
    return getrf2_impl::rec(m, n, A, incRowA, incColA, piv, incPiv);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRF2_TCC
