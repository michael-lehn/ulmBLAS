// The five variants of the LU factorization A = LU without pivoting,
// unblocked and blocked.  A is n x n, column major with leading dimension
// ld, and is overwritten with U (upper triangle) and L (strictly lower
// triangle, the unit diagonal is not stored).  Without pivoting, every
// leading principal submatrix of A must be nonsingular.
//
// The code is 0-based: the algorithms of the guide are 1-based, A(i,j) there
// is A[(i-1) + (j-1)*ld] here.
#ifndef VARIANTS_LU_H
#define VARIANTS_LU_H 1

#include <algorithm>
#include <ulmblas/ulmblas.h>

namespace lu {

// element (i, j)
inline double &
at(double *A, long ld, long i, long j)
{
    return A[i + j * ld];
}

// --- the operations of the unblocked variants on column or row j ----------

// u01 := L00^{-1} a01  (trsv, L00 unit lower triangular)
inline void
solveU01(long j, double *A, long ld)
{
    for (long k = 0; k < j; ++k) {
	const double u = at(A, ld, k, j);
	for (long i = k + 1; i < j; ++i) {
	    at(A, ld, i, j) -= at(A, ld, i, k) * u;
	}
    }
}

// l10^T := a10^T U00^{-1}  (trsv, U00 upper triangular, from the right)
inline void
solveL10(long j, double *A, long ld)
{
    for (long k = 0; k < j; ++k) {
	double s = at(A, ld, j, k);
	for (long l = 0; l < k; ++l) {
	    s -= at(A, ld, j, l) * at(A, ld, l, k);
	}
	at(A, ld, j, k) = s / at(A, ld, k, k);
    }
}

// rows j..n-1 of column j: (alpha11; a21) -= (l10^T; L20) u01  (gemv)
inline void
updateColumn(long n, long j, double *A, long ld)
{
    for (long k = 0; k < j; ++k) {
	const double u = at(A, ld, k, j);
	for (long i = j; i < n; ++i) {
	    at(A, ld, i, j) -= at(A, ld, i, k) * u;
	}
    }
}

// columns j+1..n-1 of row j: a12^T -= l10^T U02  (gemv)
inline void
updateRow(long n, long j, double *A, long ld)
{
    for (long c = j + 1; c < n; ++c) {
	double s = at(A, ld, j, c);
	for (long k = 0; k < j; ++k) {
	    s -= at(A, ld, j, k) * at(A, ld, k, c);
	}
	at(A, ld, j, c) = s;
    }
}

// l21 := a21 / mu11  (scal)
inline void
scaleColumn(long n, long j, double *A, long ld)
{
    for (long i = j + 1; i < n; ++i) {
	at(A, ld, i, j) /= at(A, ld, j, j);
    }
}

// --- unblocked ---------------------------------------------------------

// variant 1, bordered
inline void
unb1(long n, double *A, long ld)
{
    for (long j = 0; j < n; ++j) {
	solveU01(j, A, ld);            // u01 := L00^{-1} a01
	solveL10(j, A, ld);            // l10^T := a10^T U00^{-1}
	for (long k = 0; k < j; ++k) { // mu11 := alpha11 - l10^T u01
	    at(A, ld, j, j) -= at(A, ld, j, k) * at(A, ld, k, j);
	}
    }
}

// variant 2, up-looking: row j of L and of U
inline void
unb2(long n, double *A, long ld)
{
    for (long j = 0; j < n; ++j) {
	solveL10(j, A, ld);            // l10^T := a10^T U00^{-1}
	for (long k = 0; k < j; ++k) { // mu11 := alpha11 - l10^T u01
	    at(A, ld, j, j) -= at(A, ld, j, k) * at(A, ld, k, j);
	}
	updateRow(n, j, A, ld); // u12^T := a12^T - l10^T U02
    }
}

// variant 3, left-looking: column j of U and of L
inline void
unb3(long n, double *A, long ld)
{
    for (long j = 0; j < n; ++j) {
	solveU01(j, A, ld);        // u01 := L00^{-1} a01
	updateColumn(n, j, A, ld); // (alpha11; a21) -= (l10^T; L20) u01
	scaleColumn(n, j, A, ld);  // l21 := a21 / mu11
    }
}

// variant 4, Crout: row j of U and column j of L
inline void
unb4(long n, double *A, long ld)
{
    for (long j = 0; j < n; ++j) {
	updateColumn(n, j, A, ld); // (alpha11; a21) -= (l10^T; L20) u01
	updateRow(n, j, A, ld);    // u12^T := a12^T - l10^T U02
	scaleColumn(n, j, A, ld);  // l21 := a21 / mu11
    }
}

// variant 5, right-looking
inline void
unb5(long n, double *A, long ld)
{
    for (long j = 0; j < n; ++j) {
	scaleColumn(n, j, A, ld);          // l21 := a21 / mu11
	for (long c = j + 1; c < n; ++c) { // A22 := A22 - l21 u12^T  (ger)
	    const double u = at(A, ld, j, c);
	    for (long i = j + 1; i < n; ++i) {
		at(A, ld, i, c) -= at(A, ld, i, j) * u;
	    }
	}
    }
}

// --- blocked, block size nb --------------------------------------------
//
// With the blocks A00 = A(0:j, 0:j), A01 = A(0:j, j:j+b), A02 = A(0:j,
// j+b:n), A10 = A(j:j+b, 0:j), A11 = A(j:j+b, j:j+b), A12 = A(j:j+b,
// j+b:n), A20 = A(j+b:n, 0:j), A21 = A(j+b:n, j:j+b), A22 = A(j+b:n,
// j+b:n) (half-open ranges).

// X := L^{-1} X, L unit lower triangular (k x k), X k x c  (trsm)
inline void
solveLeft(long k, long c, const double *L, double *X, long ld)
{
    ulmBLAS::trlsm(k, c, 1.0, false, true, L, 1L, ld, X, 1L, ld);
}

// X := X U^{-1}, U upper triangular (k x k), X r x k  (trsm): the solve
// U^T X^T = X^T, a trlsm with the strides of U and X exchanged
inline void
solveRight(long r, long k, const double *U, double *X, long ld)
{
    ulmBLAS::trlsm(k, r, 1.0, false, false, U, ld, 1L, X, ld, 1L);
}

// C := C - A B, C r x c, A r x k, B k x c  (gemm)
inline void
update(long r, long c, long k, const double *A_, const double *B, double *C,
       long ld)
{
    ulmBLAS::gemm(r, c, k, -1.0, false, A_, 1L, ld, false, B, 1L, ld, 1.0, C,
                  1L, ld);
}

// the blocks of the step at row and column j with block size b
struct Blocks
{
    double *A00, *A01, *A02, *A10, *A11, *A12, *A20, *A21, *A22;
    long b, n2; // A11 is b x b, A22 is n2 x n2

    Blocks(long n, double *A, long ld, long j, long nb)
        : b(std::min(nb, n - j)), n2(n - j - b)
    {
	A00 = &at(A, ld, 0, 0);
	A01 = &at(A, ld, 0, j);
	A02 = &at(A, ld, 0, j + b);
	A10 = &at(A, ld, j, 0);
	A11 = &at(A, ld, j, j);
	A12 = &at(A, ld, j, j + b);
	A20 = &at(A, ld, j + b, 0);
	A21 = &at(A, ld, j + b, j);
	A22 = &at(A, ld, j + b, j + b);
    }
};

// variant 1, bordered
inline void
blk1(long n, double *A, long ld, long nb)
{
    for (long j = 0; j < n; j += nb) {
	const Blocks B(n, A, ld, j, nb);
	solveLeft(j, B.b, B.A00, B.A01, ld);          // A01 := L00^{-1} A01
	solveRight(B.b, j, B.A00, B.A10, ld);         // A10 := A10 U00^{-1}
	update(B.b, B.b, j, B.A10, B.A01, B.A11, ld); // A11 -= A10 A01
	unb1(B.b, B.A11, ld);                         // A11 := LU(A11)
    }
}

// variant 2, up-looking
inline void
blk2(long n, double *A, long ld, long nb)
{
    for (long j = 0; j < n; j += nb) {
	const Blocks B(n, A, ld, j, nb);
	solveRight(B.b, j, B.A00, B.A10, ld);          // A10 := A10 U00^{-1}
	update(B.b, B.b, j, B.A10, B.A01, B.A11, ld);  // A11 -= A10 A01
	update(B.b, B.n2, j, B.A10, B.A02, B.A12, ld); // A12 -= A10 A02
	unb2(B.b, B.A11, ld);                          // A11 := LU(A11)
	solveLeft(B.b, B.n2, B.A11, B.A12, ld);        // A12 := L11^{-1} A12
    }
}

// variant 3, left-looking
inline void
blk3(long n, double *A, long ld, long nb)
{
    for (long j = 0; j < n; j += nb) {
	const Blocks B(n, A, ld, j, nb);
	solveLeft(j, B.b, B.A00, B.A01, ld);           // A01 := L00^{-1} A01
	update(B.b, B.b, j, B.A10, B.A01, B.A11, ld);  // A11 -= A10 A01
	update(B.n2, B.b, j, B.A20, B.A01, B.A21, ld); // A21 -= A20 A01
	unb3(B.b, B.A11, ld);                          // A11 := LU(A11)
	solveRight(B.n2, B.b, B.A11, B.A21, ld);       // A21 := A21 U11^{-1}
    }
}

// variant 4, Crout
inline void
blk4(long n, double *A, long ld, long nb)
{
    for (long j = 0; j < n; j += nb) {
	const Blocks B(n, A, ld, j, nb);
	update(B.b, B.b, j, B.A10, B.A01, B.A11, ld);  // A11 -= A10 A01
	update(B.b, B.n2, j, B.A10, B.A02, B.A12, ld); // A12 -= A10 A02
	update(B.n2, B.b, j, B.A20, B.A01, B.A21, ld); // A21 -= A20 A01
	unb4(B.b, B.A11, ld);                          // A11 := LU(A11)
	solveLeft(B.b, B.n2, B.A11, B.A12, ld);        // A12 := L11^{-1} A12
	solveRight(B.n2, B.b, B.A11, B.A21, ld);       // A21 := A21 U11^{-1}
    }
}

// variant 5, right-looking
inline void
blk5(long n, double *A, long ld, long nb)
{
    for (long j = 0; j < n; j += nb) {
	const Blocks B(n, A, ld, j, nb);
	unb5(B.b, B.A11, ld);                             // A11 := LU(A11)
	solveLeft(B.b, B.n2, B.A11, B.A12, ld);           // A12 := L11^{-1} A12
	solveRight(B.n2, B.b, B.A11, B.A21, ld);          // A21 := A21 U11^{-1}
	update(B.n2, B.n2, B.b, B.A21, B.A12, B.A22, ld); // A22 -= A21 A12
    }
}

} // namespace lu

#endif // VARIANTS_LU_H
