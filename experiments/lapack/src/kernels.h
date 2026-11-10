#ifndef ULMLAPACK_KERNELS_H
#define ULMLAPACK_KERNELS_H 1

//
//  Serial building blocks of the task-based factorizations.  They work on
//  matrices with a row and a column stride (as everything in ulmBLAS), so a
//  transposed view costs nothing: the upper Cholesky factorization is the
//  lower one of the transposed view, a solve from the right is a solve from
//  the left with the transposed right-hand side.
//
//  The numerical work is done by ulmBLAS (gemm, trlsm, sylrk, getf2); only
//  the unblocked Cholesky kernel and the row interchanges are loops here.
//

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <ulmblas/ulmblas.h>
#include <ulmblas/lapack/getf2.h>

namespace ulmlapack {

typedef std::ptrdiff_t Index;

//  Base case sizes of the recursive algorithms (compile time parameters
//  for the tuning, see README).
#ifndef ULMLAPACK_POTRF_BASE
#define ULMLAPACK_POTRF_BASE 64
#endif
#ifndef ULMLAPACK_GETRF_BASE
#define ULMLAPACK_GETRF_BASE 16
#endif
#ifndef ULMLAPACK_GETRF_BASE_SMALL
#define ULMLAPACK_GETRF_BASE_SMALL 32 // for m <= 96
#endif
const Index POTRF_BASE = ULMLAPACK_POTRF_BASE;
const Index GETRF_BASE = ULMLAPACK_GETRF_BASE;
const Index GETRF_BASE_SMALL = ULMLAPACK_GETRF_BASE_SMALL;

//
//  Unblocked Cholesky factorization A = L L^T of the lower triangle of an
//  n x n matrix.  Returns 0 or j+1 if the j-th pivot (0-based) is not
//  positive or NaN (as LAPACK's dpotf2).  Only the lower triangle is
//  referenced.
//
inline Index
potf2_lower(Index n, double *A, Index rs, Index cs)
{
    for (Index j = 0; j < n; ++j) {
	double ajj = A[j * (rs + cs)];
	if (!(ajj > 0)) { // also NaN
	    return j + 1;
	}
	ajj = std::sqrt(ajj);
	A[j * (rs + cs)] = ajj;
	const double r = 1.0 / ajj;
	double *a = &A[j * cs];
	for (Index i = j + 1; i < n; ++i) {
	    a[i * rs] *= r;
	}
	// rank-1 update of the lower triangle of A(j+1:n, j+1:n)
	for (Index l = j + 1; l < n; ++l) {
	    const double alj = a[l * rs];
	    double *c = &A[l * cs];
	    for (Index i = l; i < n; ++i) {
		c[i * rs] -= a[i * rs] * alj;
	    }
	}
    }
    return 0;
}

//
//  The same, left-looking (Crout): column j is updated with the columns
//  0..j-1 (A(j:n,j) -= A(j:n,0:j) A(j,0:j)^T), four at a time, then scaled.
//  The target column is loaded and stored once per four columns instead of
//  once per column as in the right-looking version; for unit row stride the
//  inner loops are contiguous and vectorized by the compiler.
//
//  Row by row (Cholesky-Banachiewicz) for unit column stride, i.e. the
//  transposed view of the upper variant: L(i,j) = (A(i,j) - L(i,0:j)
//  L(j,0:j)^T) / L(j,j), dot products over contiguous rows with four
//  partial sums.
inline Index
potf2_lower_rows(Index n, double *A, Index rs)
{
    for (Index i = 0; i < n; ++i) {
	double *li = &A[i * rs];
	for (Index j = 0; j <= i; ++j) {
	    const double *lj = &A[j * rs];
	    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
	    Index k = 0;
	    for (; k + 4 <= j; k += 4) {
		s0 += li[k] * lj[k];
		s1 += li[k + 1] * lj[k + 1];
		s2 += li[k + 2] * lj[k + 2];
		s3 += li[k + 3] * lj[k + 3];
	    }
	    for (; k < j; ++k) {
		s0 += li[k] * lj[k];
	    }
	    const double s = li[j] - ((s0 + s1) + (s2 + s3));
	    if (j < i) {
		li[j] = s / lj[j];
	    } else {
		if (!(s > 0)) { // also NaN
		    return i + 1;
		}
		li[i] = std::sqrt(s);
	    }
	}
    }
    return 0;
}

inline Index
potf2_lower_crout(Index n, double *A, Index rs, Index cs)
{
    if (rs != 1) {
	return (cs == 1) ? potf2_lower_rows(n, A, rs)
	                 : potf2_lower(n, A, rs, cs);
    }
    for (Index j = 0; j < n; ++j) {
	double *c = &A[j * cs];
	Index k = 0;
	for (; k + 4 <= j; k += 4) {
	    const double *a0 = &A[(k + 0) * cs], *a1 = &A[(k + 1) * cs];
	    const double *a2 = &A[(k + 2) * cs], *a3 = &A[(k + 3) * cs];
	    const double b0 = a0[j], b1 = a1[j], b2 = a2[j], b3 = a3[j];
	    for (Index i = j; i < n; ++i) {
		c[i] -= a0[i] * b0 + a1[i] * b1 + a2[i] * b2 + a3[i] * b3;
	    }
	}
	for (; k < j; ++k) {
	    const double *a0 = &A[k * cs];
	    const double b0 = a0[j];
	    for (Index i = j; i < n; ++i) {
		c[i] -= a0[i] * b0;
	    }
	}
	double cjj = c[j];
	if (!(cjj > 0)) { // also NaN
	    return j + 1;
	}
	cjj = std::sqrt(cjj);
	c[j] = cjj;
	const double r = 1.0 / cjj;
	for (Index i = j + 1; i < n; ++i) {
	    c[i] *= r;
	}
    }
    return 0;
}

//
//  Recursive Cholesky factorization of the lower triangle (as LAPACK's
//  dpotrf2, with an unblocked base case):
//
//      [ A11     ]   [ L11     ] [ L11^T L21^T ]
//      [ A21 A22 ] = [ L21 L22 ] [       L22^T ]
//
//  L11 = chol(A11),  L21 = A21 L11^{-T},  L22 = chol(A22 - L21 L21^T).
//
inline Index
potrf_lower_rec(Index n, double *A, Index rs, Index cs)
{
    if (n <= POTRF_BASE) {
	return potf2_lower_crout(n, A, rs, cs);
    }
    const Index n1 = n / 2;
    const Index n2 = n - n1;

    Index info = potrf_lower_rec(n1, A, rs, cs);
    if (info) {
	return info;
    }
    double *A21 = &A[n1 * rs];
    double *A22 = &A[n1 * (rs + cs)];

    // A21 L11^T = A21  <=>  L11 A21^T = A21^T
    ulmBLAS::trlsm(n1, n2, 1.0, false, false, A, rs, cs, A21, cs, rs);
    ulmBLAS::sylrk(n2, n1, -1.0, A21, rs, cs, 1.0, A22, rs, cs);

    info = potrf_lower_rec(n2, A22, rs, cs);
    return info ? info + n1 : 0;
}

//
//  Row interchanges of LAPACK's dlaswp (forward, increment 1) on n columns:
//  for i = k1, ..., k2-1 rows i and piv[i] are swapped.  piv is 0-based and
//  relative to A.  Column by column, which is cache friendly for column
//  major storage.
//
inline void
laswp(Index n, double *A, Index rs, Index cs, Index k1, Index k2,
      const Index *piv)
{
    if (rs == 1) {
	for (Index j = 0; j < n; ++j) {
	    double *a = &A[j * cs];
	    for (Index i = k1; i < k2; ++i) {
		const Index p = piv[i];
		if (p != i) {
		    std::swap(a[i], a[p]);
		}
	    }
	}
    } else {
	for (Index j = 0; j < n; ++j) {
	    double *a = &A[j * cs];
	    for (Index i = k1; i < k2; ++i) {
		const Index p = piv[i];
		if (p != i) {
		    std::swap(a[i * rs], a[p * rs]);
		}
	    }
	}
    }
}

//
//  The unblocked kernels are fast if one of the strides is 1 (column major,
//  row major, or the transposed views).  If neither is, the serial
//  factorizations below run on a contiguous column-major copy, like the
//  BLAS kernels pack their operands: O(n^2) copies for O(n^3) operations,
//  and only of the block the task owns, never of the whole matrix.
//
//  lowerOnly: only the lower triangle is copied (and written back), the
//  other one is not referenced (Cholesky).
//
template <typename F>
Index
withUnitStride(Index m, Index n, double *A, Index rs, Index cs, bool lowerOnly,
               F f)
{
    if (rs == 1 || cs == 1) {
	return f(A, rs, cs);
    }
    std::vector<double> B(std::size_t(m) * n);
    for (Index j = 0; j < n; ++j) {
	for (Index i = lowerOnly ? j : 0; i < m; ++i) {
	    B[i + j * m] = A[i * rs + j * cs];
	}
    }
    const Index info = f(B.data(), Index(1), m);
    for (Index j = 0; j < n; ++j) {
	for (Index i = lowerOnly ? j : 0; i < m; ++i) {
	    A[i * rs + j * cs] = B[i + j * m];
	}
    }
    return info;
}

inline Index
potrf_lower_packed(Index n, double *A, Index rs, Index cs)
{
    return withUnitStride(n, n, A, rs, cs, true,
                          [n](double *B, Index r, Index c) {
	                      return potrf_lower_rec(n, B, r, c);
                          });
}

//
//  Recursive LU factorization with partial pivoting of an m x n matrix (as
//  LAPACK's dgetrf2; the base case is ulmBLAS's unblocked getf2).  piv is
//  0-based and relative to A, min(m,n) entries.  Returns 0 or j+1 for the
//  first exactly zero pivot U(j,j); the factorization is completed in any
//  case.
//
inline Index
getrf_rec(Index m, Index n, double *A, Index rs, Index cs, Index *piv)
{
    const Index kmin = std::min(m, n);
    if (kmin == 0) {
	return 0;
    }
    if (n <= (m <= 96 ? GETRF_BASE_SMALL : GETRF_BASE) || m == 1) {
	return ulmBLAS::getf2(m, n, A, rs, cs, piv, Index(1));
    }
    const Index n1 = kmin / 2;
    const Index n2 = n - n1;

    Index info = getrf_rec(m, n1, A, rs, cs, piv);

    double *A12 = &A[n1 * cs];
    double *A21 = &A[n1 * rs];
    double *A22 = &A[n1 * (rs + cs)];

    laswp(n2, A12, rs, cs, 0, n1, piv);
    ulmBLAS::trlsm(n1, n2, 1.0, false, true, A, rs, cs, A12, rs, cs);
    if (m > n1) {
	ulmBLAS::gemm(m - n1, n2, n1, -1.0, false, A21, rs, cs, false, A12, rs,
	              cs, 1.0, A22, rs, cs);
    }

    Index info2 = getrf_rec(m - n1, n2, A22, rs, cs, piv + n1);
    if (info == 0 && info2 > 0) {
	info = info2 + n1;
    }
    for (Index i = n1; i < kmin; ++i) {
	piv[i] += n1;
    }
    laswp(n1, A, rs, cs, n1, kmin, piv);
    return info;
}

inline Index
getrf_packed(Index m, Index n, double *A, Index rs, Index cs, Index *piv)
{
    return withUnitStride(m, n, A, rs, cs, false,
                          [m, n, piv](double *B, Index r, Index c) {
	                      return getrf_rec(m, n, B, r, c, piv);
                          });
}

} // namespace ulmlapack

#endif // ULMLAPACK_KERNELS_H
