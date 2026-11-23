// The three variants of the Cholesky factorization A = L L^T, unblocked and
// blocked.  A is n x n, column major with leading dimension ld; only the
// lower triangle is read and overwritten with L.  The result is info: 0, or
// j + 1 if the j-th diagonal element (0-based) is not positive.
//
// The code is 0-based: the algorithms of the guide are 1-based, A(i,j) there
// is A[(i-1) + (j-1)*ld] here.
#ifndef VARIANTS_CHOL_H
#define VARIANTS_CHOL_H 1

#include <algorithm>
#include <cmath>
#include <ulmblas/ulmblas.h>

namespace chol {

// element (i, j)
inline double &
at(double *A, long ld, long i, long j)
{
    return A[i + j * ld];
}

// --- unblocked ---------------------------------------------------------

// variant 1, bordered: row j of L from the rows above
inline long
unb1(long n, double *A, long ld)
{
    for (long j = 0; j < n; ++j) {
	// l10^T := a10^T L00^{-T}  (trsv)
	for (long k = 0; k < j; ++k) {
	    double s = at(A, ld, j, k);
	    for (long l = 0; l < k; ++l) {
		s -= at(A, ld, j, l) * at(A, ld, k, l);
	    }
	    at(A, ld, j, k) = s / at(A, ld, k, k);
	}
	// lambda11 := sqrt(alpha11 - l10^T l10)  (dot)
	double d = at(A, ld, j, j);
	for (long k = 0; k < j; ++k) {
	    d -= at(A, ld, j, k) * at(A, ld, j, k);
	}
	if (!(d > 0)) {
	    return j + 1;
	}
	at(A, ld, j, j) = std::sqrt(d);
    }
    return 0;
}

// variant 2, left-looking: column j of L from the columns to its left
inline long
unb2(long n, double *A, long ld)
{
    for (long j = 0; j < n; ++j) {
	// (alpha11; a21) -= (l10^T; L20) l10  (gemv, one axpy per column k)
	for (long k = 0; k < j; ++k) {
	    const double b = at(A, ld, j, k);
	    for (long i = j; i < n; ++i) {
		at(A, ld, i, j) -= at(A, ld, i, k) * b;
	    }
	}
	// lambda11 := sqrt(alpha11), l21 := a21 / lambda11  (scal)
	if (!(at(A, ld, j, j) > 0)) {
	    return j + 1;
	}
	at(A, ld, j, j) = std::sqrt(at(A, ld, j, j));
	for (long i = j + 1; i < n; ++i) {
	    at(A, ld, i, j) /= at(A, ld, j, j);
	}
    }
    return 0;
}

// variant 3, right-looking: column j of L, then the update of A22
inline long
unb3(long n, double *A, long ld)
{
    for (long j = 0; j < n; ++j) {
	// lambda11 := sqrt(alpha11), l21 := a21 / lambda11  (scal)
	if (!(at(A, ld, j, j) > 0)) {
	    return j + 1;
	}
	at(A, ld, j, j) = std::sqrt(at(A, ld, j, j));
	for (long i = j + 1; i < n; ++i) {
	    at(A, ld, i, j) /= at(A, ld, j, j);
	}
	// A22 := A22 - l21 l21^T, lower triangle  (syr)
	for (long k = j + 1; k < n; ++k) {
	    const double b = at(A, ld, k, j);
	    for (long i = k; i < n; ++i) {
		at(A, ld, i, k) -= at(A, ld, i, j) * b;
	    }
	}
    }
    return 0;
}

// --- blocked, block size nb --------------------------------------------
//
// With the blocks A00 = A(0:j, 0:j), A10 = A(j:j+b, 0:j), A11 = A(j:j+b,
// j:j+b), A20 = A(j+b:n, 0:j), A21 = A(j+b:n, j:j+b), A22 = A(j+b:n,
// j+b:n) (half-open ranges).  X := X L^{-T} is the triangular solve
// L X^T = X^T, a trlsm with the strides of X exchanged.

// variant 1, bordered
inline long
blk1(long n, double *A, long ld, long nb)
{
    for (long j = 0; j < n; j += nb) {
	const long b = std::min(nb, n - j);
	double *A00 = &at(A, ld, 0, 0), *A10 = &at(A, ld, j, 0),
	       *A11 = &at(A, ld, j, j);
	// A10 := A10 L00^{-T}  (trsm)
	ulmBLAS::trlsm(j, b, 1.0, false, false, A00, 1L, ld, A10, ld, 1L);
	// A11 := A11 - A10 A10^T  (syrk)
	ulmBLAS::sylrk(b, j, -1.0, A10, 1L, ld, 1.0, A11, 1L, ld);
	// A11 := Chol(A11)
	if (const long info = unb1(b, A11, ld)) {
	    return j + info;
	}
    }
    return 0;
}

// variant 2, left-looking
inline long
blk2(long n, double *A, long ld, long nb)
{
    for (long j = 0; j < n; j += nb) {
	const long b = std::min(nb, n - j), m2 = n - j - b;
	double *A10 = &at(A, ld, j, 0), *A11 = &at(A, ld, j, j),
	       *A20 = &at(A, ld, j + b, 0), *A21 = &at(A, ld, j + b, j);
	// A11 := A11 - A10 A10^T  (syrk)
	ulmBLAS::sylrk(b, j, -1.0, A10, 1L, ld, 1.0, A11, 1L, ld);
	// A21 := A21 - A20 A10^T  (gemm)
	ulmBLAS::gemm(m2, b, j, -1.0, false, A20, 1L, ld, false, A10, ld, 1L,
	              1.0, A21, 1L, ld);
	// A11 := Chol(A11), A21 := A21 L11^{-T}  (trsm)
	if (const long info = unb2(b, A11, ld)) {
	    return j + info;
	}
	ulmBLAS::trlsm(b, m2, 1.0, false, false, A11, 1L, ld, A21, ld, 1L);
    }
    return 0;
}

// variant 3, right-looking
inline long
blk3(long n, double *A, long ld, long nb)
{
    for (long j = 0; j < n; j += nb) {
	const long b = std::min(nb, n - j), m2 = n - j - b;
	double *A11 = &at(A, ld, j, j), *A21 = &at(A, ld, j + b, j),
	       *A22 = &at(A, ld, j + b, j + b);
	// A11 := Chol(A11), A21 := A21 L11^{-T}  (trsm)
	if (const long info = unb3(b, A11, ld)) {
	    return j + info;
	}
	ulmBLAS::trlsm(b, m2, 1.0, false, false, A11, 1L, ld, A21, ld, 1L);
	// A22 := A22 - A21 A21^T  (syrk)
	ulmBLAS::sylrk(m2, b, -1.0, A21, 1L, ld, 1.0, A22, 1L, ld);
    }
    return 0;
}

} // namespace chol

#endif // VARIANTS_CHOL_H
