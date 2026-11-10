#ifndef ULMBLAS_LAPACK_GETRF2_TCC
#define ULMBLAS_LAPACK_GETRF2_TCC 1

#include <algorithm>
#include <utility>
#include <ulmblas/lapack/islapacktype.h>
#include <ulmblas/lapack/buffer.h>
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
//  Row interchanges (forward): for i = k1, ..., k2-1 rows i and piv[i] of
//  the n columns of A are swapped.  piv is 0-based and contiguous.  Column
//  by column, which is cache friendly for column-major storage.
//
template <typename IndexType, typename T>
void
swapRows(IndexType n, T *A, IndexType rs, IndexType cs, IndexType k1,
         IndexType k2, const IndexType *piv)
{
    for (IndexType j = 0; j < n; ++j) {
	T *a = &A[j * cs];
	for (IndexType i = k1; i < k2; ++i) {
	    const IndexType p = piv[i];
	    if (p != i) {
		std::swap(a[i * rs], a[p * rs]);
	    }
	}
    }
}

template <typename IndexType, typename T>
IndexType
rec(IndexType m, IndexType n, T *A, IndexType rs, IndexType cs, IndexType *piv)
{
    const IndexType kmin = std::min(m, n);
    if (kmin == 0) {
	return 0;
    }
    if (n <= (m <= 96 ? ULMBLAS_GETRF2_BASE_SMALL : ULMBLAS_GETRF2_BASE) ||
        m == 1) {
	return getf2(m, n, A, rs, cs, piv, IndexType(1));
    }
    const IndexType n1 = kmin / 2;
    const IndexType n2 = n - n1;

    IndexType info = rec(m, n1, A, rs, cs, piv);

    T *A12 = &A[n1 * cs];
    T *A21 = &A[n1 * rs];
    T *A22 = &A[n1 * (rs + cs)];

    swapRows(n2, A12, rs, cs, IndexType(0), n1, piv);
    trlsm(n1, n2, T(1), false, true, A, rs, cs, A12, rs, cs);
    if (m > n1) {
	gemm(m - n1, n2, n1, T(-1), false, A21, rs, cs, false, A12, rs, cs,
	     T(1), A22, rs, cs);
    }

    IndexType info2 = rec(m - n1, n2, A22, rs, cs, piv + n1);
    if (info == 0 && info2 > 0) {
	info = info2 + n1;
    }
    for (IndexType i = n1; i < kmin; ++i) {
	piv[i] += n1;
    }
    swapRows(n1, A, rs, cs, n1, kmin, piv);
    return info;
}

} // namespace getrf2_impl

template <typename IndexType, typename T, typename PivIndex>
IndexType
getrf2(IndexType m, IndexType n, T *A, IndexType incRowA, IndexType incColA,
       PivIndex *piv, IndexType incPiv)
{
    ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T);

    const IndexType kmin = std::min(m, n);
    if (kmin <= 0) {
	return 0;
    }
    Buffer<IndexType> p(kmin);
    IndexType *pp = p.data();
    const IndexType info =
        withUnitStride(m, n, A, incRowA, incColA, false,
	               [m, n, pp](T *B, IndexType rs, IndexType cs) {
	                   return getrf2_impl::rec(m, n, B, rs, cs, pp);
                       });
    for (IndexType i = 0; i < kmin; ++i) {
	piv[i * incPiv] = PivIndex(p[i]);
    }
    return info;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRF2_TCC
