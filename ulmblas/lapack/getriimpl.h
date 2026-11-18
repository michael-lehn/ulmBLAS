#ifndef ULMBLAS_LAPACK_GETRIIMPL_H
#define ULMBLAS_LAPACK_GETRIIMPL_H 1

//
//  The product A := U L in place of getri (serial), used by getri and by
//  the task version (tasks/inversetasks.tcc).  A header of its own, as
//  lauumimpl.h.
//

#include <ulmblas/level3/gemm.h>
#include <ulmblas/level3/trumm.h>

//  base case size (compile time parameter)
#ifndef ULMBLAS_GETRI_BASE
#define ULMBLAS_GETRI_BASE 32
#endif

namespace ulmBLAS {

namespace getri_impl {

//
//  A := U L in place, U upper triangular (upper triangle and diagonal of
//  A), L unit lower triangular (strictly lower triangle of A), unblocked.
//  Step i computes row and column i of the product: (U L)(i,j) needs U(i,k)
//  and L(k,j) for k >= max(i,j) only, so rows and columns < i (already the
//  product) are not referenced.
//
template <typename IndexType, typename T>
void
ul2(IndexType n, T *A, IndexType rs, IndexType cs)
{
    for (IndexType i = 0; i < n; ++i) {
	T *Ai = &A[i * rs]; // row i
	T *ai = &A[i * cs]; // column i

	T d = Ai[i * cs];
	for (IndexType k = i + 1; k < n; ++k) {
	    d += Ai[k * cs] * ai[k * rs];
	}
	Ai[i * cs] = d;
	//  row: X(i,j) = U(i,j) + sum_{k>j} U(i,k) L(k,j), j increasing
	for (IndexType j = i + 1; j < n; ++j) {
	    T s = Ai[j * cs];
	    for (IndexType k = j + 1; k < n; ++k) {
		s += Ai[k * cs] * A[k * rs + j * cs];
	    }
	    Ai[j * cs] = s;
	}
	//  column: X(j,i) = sum_{k>=j} U(j,k) L(k,i), j increasing
	for (IndexType j = i + 1; j < n; ++j) {
	    T s = A[j * (rs + cs)] * ai[j * rs];
	    for (IndexType k = j + 1; k < n; ++k) {
		s += A[j * rs + k * cs] * ai[k * rs];
	    }
	    ai[j * rs] = s;
	}
    }
}

//  A := A L, L unit lower triangular (m x m), A n x m
template <typename IndexType, typename T>
void
trmmRightLowerUnit(IndexType m, IndexType n, const T *L, IndexType rs,
                   IndexType cs, T *A, IndexType ars, IndexType acs)
{
    //  A^T := L^T A^T, L^T unit upper (transposed view)
    trumm(m, n, T(1), false, true, L, cs, rs, A, acs, ars);
}

//
//  [U11 U12] [L11    ]   [U11 L11 + U12 L21  U12 L22]
//  [    U22] [L21 L22] = [U22 L21            U22 L22]
//
template <typename IndexType, typename T>
void
rec(IndexType n, T *A, IndexType rs, IndexType cs)
{
    if (n <= ULMBLAS_GETRI_BASE) {
	ul2(n, A, rs, cs);
	return;
    }
    const IndexType n1 = n / 2, n2 = n - n1;
    T *A12 = &A[n1 * cs];
    T *A21 = &A[n1 * rs];
    T *A22 = &A[n1 * (rs + cs)];

    rec(n1, A, rs, cs);
    gemm(n1, n1, n2, T(1), false, A12, rs, cs, false, A21, rs, cs, T(1), A, rs,
         cs);
    //  A12 := A12 L22
    trmmRightLowerUnit(n2, n1, A22, rs, cs, A12, rs, cs);
    //  A21 := U22 A21
    trumm(n2, n1, T(1), false, false, A22, rs, cs, A21, rs, cs);
    rec(n2, A22, rs, cs);
}

} // namespace getri_impl

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRIIMPL_H
