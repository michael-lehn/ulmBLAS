// The three variants of the LU factorization with partial pivoting, PA = LU,
// unblocked and blocked: variants 3, 4 and 5 of lu.h, the ones that have
// the updated column j below the diagonal when they factor it.  A is m x n
// with m >= n (unblocked: the panels of the blocked variants) or n x n
// (blocked), column major with leading dimension ld.  piv[j] is the row
// interchanged with row j in step j (0-based, as in LAPACK).  The result is
// info: 0, or j + 1 for the first column j without a nonzero pivot.
//
// The code is 0-based: the algorithms of the guide are 1-based, A(i,j) there
// is A[(i-1) + (j-1)*ld] here.
#ifndef VARIANTS_LUPIV_H
#define VARIANTS_LUPIV_H 1

#include "lu.h"
#include <cmath>
#include <ulmblas/lapack/laswp.h>
#include <utility>

namespace lupiv {

using lu::at;

// the pivot of column j: the row of the largest |A(i,j)|, i >= j
inline long
pivot(long m, long j, double *A, long ld)
{
    long p = j;
    for (long i = j + 1; i < m; ++i) {
	if (std::abs(at(A, ld, i, j)) > std::abs(at(A, ld, p, j))) {
	    p = i;
	}
    }
    return p;
}

// interchange rows j and p in columns c0..c1-1
inline void
swapRows(long j, long p, long c0, long c1, double *A, long ld)
{
    if (p != j) {
	for (long c = c0; c < c1; ++c) {
	    std::swap(at(A, ld, j, c), at(A, ld, p, c));
	}
    }
}

// chooses the pivot of column j and interchanges the rows in columns
// c0..c1-1; then l21 := a21 / mu11, unless the pivot is zero
inline void
pivotAndScale(long m, long j, long c0, long c1, double *A, long ld, long *piv,
              long &info)
{
    piv[j] = pivot(m, j, A, ld);
    swapRows(j, piv[j], c0, c1, A, ld);
    if (at(A, ld, j, j) != 0) {
	lu::scaleColumn(m, j, A, ld);
    } else if (!info) {
	info = j + 1;
    }
}

// --- unblocked, A m x n, m >= n ------------------------------------------

// variant 3, left-looking: the interchanges reach column j when it is
// factored
inline long
unb3(long m, long n, double *A, long ld, long *piv)
{
    long info = 0;
    for (long j = 0; j < n; ++j) {
	for (long i = 0; i < j; ++i) { // the earlier interchanges, column j
	    std::swap(at(A, ld, i, j), at(A, ld, piv[i], j));
	}
	lu::solveU01(j, A, ld);        // u01 := L00^{-1} a01
	lu::updateColumn(m, j, A, ld); // (alpha11; a21) -= (l10^T; L20) u01
	pivotAndScale(m, j, 0, j + 1, A, ld, piv, info); // columns 0..j
    }
    return info;
}

// variant 4, Crout: the interchanges reach all columns at once
inline long
unb4(long m, long n, double *A, long ld, long *piv)
{
    long info = 0;
    for (long j = 0; j < n; ++j) {
	lu::updateColumn(m, j, A, ld); // (alpha11; a21) -= (l10^T; L20) u01
	piv[j] = pivot(m, j, A, ld);
	swapRows(j, piv[j], 0, n, A, ld);
	lu::updateRow(n, j, A, ld); // u12^T := a12^T - l10^T U02
	if (at(A, ld, j, j) != 0) {
	    lu::scaleColumn(m, j, A, ld); // l21 := a21 / mu11
	} else if (!info) {
	    info = j + 1;
	}
    }
    return info;
}

// variant 5, right-looking
inline long
unb5(long m, long n, double *A, long ld, long *piv)
{
    long info = 0;
    for (long j = 0; j < n; ++j) {
	pivotAndScale(m, j, 0, n, A, ld, piv, info);
	for (long c = j + 1; c < n; ++c) { // A22 := A22 - l21 u12^T  (ger)
	    const double u = at(A, ld, j, c);
	    for (long i = j + 1; i < m; ++i) {
		at(A, ld, i, c) -= at(A, ld, i, j) * u;
	    }
	}
    }
    return info;
}

// --- blocked, A n x n, block size nb ---------------------------------------
//
// The panel (A11; A21) is factored by the unblocked variant of the same
// number; its interchanges, relative to row j, are made absolute and applied
// to further columns by laswp.

// apply the interchanges of rows j0..j1-1 to the columns c0..c1-1
inline void
interchange(long j0, long j1, long c0, long c1, double *A, long ld,
            const long *piv)
{
    ulmBLAS::laswp(c1 - c0, &at(A, ld, 0, c0), 1L, ld, j0, j1, piv, 1L);
}

// factor the panel of the columns j..j+b-1 by unb, pivots made absolute
template <typename Unb>
long
panel(long n, long j, long b, double *A, long ld, long *piv, Unb unb)
{
    const long info = unb(n - j, b, &at(A, ld, j, j), ld, &piv[j]);
    for (long i = j; i < j + b; ++i) {
	piv[i] += j;
    }
    return info ? info + j : 0;
}

// variant 3, left-looking
inline long
blk3(long n, double *A, long ld, long *piv, long nb)
{
    long info = 0;
    for (long j = 0; j < n; j += nb) {
	const lu::Blocks B(n, A, ld, j, nb);
	interchange(0, j, j, j + B.b, A, ld, piv); // earlier ones, the panel
	lu::solveLeft(j, B.b, B.A00, B.A01, ld);   // A01 := L00^{-1} A01
	// (A11; A21) -= (A10; A20) A01
	lu::update(n - j, B.b, j, B.A10, B.A01, B.A11, ld);
	const long i = panel(n, j, B.b, A, ld, piv, unb3);
	info = info ? info : i;
	interchange(j, j + B.b, 0, j, A, ld, piv); // to the left
    }
    return info;
}

// variant 4, Crout
inline long
blk4(long n, double *A, long ld, long *piv, long nb)
{
    long info = 0;
    for (long j = 0; j < n; j += nb) {
	const lu::Blocks B(n, A, ld, j, nb);
	// (A11; A21) -= (A10; A20) A01
	lu::update(n - j, B.b, j, B.A10, B.A01, B.A11, ld);
	const long i = panel(n, j, B.b, A, ld, piv, unb4);
	info = info ? info : i;
	interchange(j, j + B.b, 0, j, A, ld, piv);         // to the left
	interchange(j, j + B.b, j + B.b, n, A, ld, piv);   // to the right
	lu::update(B.b, B.n2, j, B.A10, B.A02, B.A12, ld); // A12 -= A10 A02
	lu::solveLeft(B.b, B.n2, B.A11, B.A12, ld); // A12 := L11^{-1} A12
    }
    return info;
}

// variant 5, right-looking
inline long
blk5(long n, double *A, long ld, long *piv, long nb)
{
    long info = 0;
    for (long j = 0; j < n; j += nb) {
	const lu::Blocks B(n, A, ld, j, nb);
	const long i = panel(n, j, B.b, A, ld, piv, unb5);
	info = info ? info : i;
	interchange(j, j + B.b, 0, j, A, ld, piv);       // to the left
	interchange(j, j + B.b, j + B.b, n, A, ld, piv); // to the right
	lu::solveLeft(B.b, B.n2, B.A11, B.A12, ld);      // A12 := L11^{-1} A12
	lu::update(B.n2, B.n2, B.b, B.A21, B.A12, B.A22, ld); // A22 -= A21 A12
    }
    return info;
}

} // namespace lupiv

#endif // VARIANTS_LUPIV_H
