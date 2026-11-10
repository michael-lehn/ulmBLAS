#ifndef ULMBLAS_LAPACK_POTF2_TCC
#define ULMBLAS_LAPACK_POTF2_TCC 1

#include <cmath>
#include <complex>
#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/lapack/islapacktype.h>
#include <ulmblas/lapack/potf2.h>

namespace ulmBLAS {

namespace potf2_impl {

//
//  Right-looking, any strides: after column j is scaled, the rest of the
//  lower triangle gets the rank-1 update.
//
template <typename IndexType, typename T>
IndexType
rightLooking(IndexType n, T *A, IndexType rs, IndexType cs)
{
    typedef decltype(std::real(T())) Real;

    for (IndexType j = 0; j < n; ++j) {
	Real ajj = std::real(A[j * (rs + cs)]);
	if (!(ajj > 0)) { // also NaN
	    return j + 1;
	}
	ajj = std::sqrt(ajj);
	A[j * (rs + cs)] = ajj;
	const Real r = Real(1) / ajj;
	T *a = &A[j * cs];
	for (IndexType i = j + 1; i < n; ++i) {
	    a[i * rs] *= r;
	}
	for (IndexType l = j + 1; l < n; ++l) {
	    const T alj = conjugate(a[l * rs]);
	    T *c = &A[l * cs];
	    for (IndexType i = l; i < n; ++i) {
		c[i * rs] -= a[i * rs] * alj;
	    }
	}
    }
    return 0;
}

//
//  Left-looking (Crout) for unit row stride: column j is updated with the
//  columns 0..j-1, four at a time, then scaled.  The target column is
//  loaded and stored once per four columns; the inner loops are contiguous.
//
template <typename IndexType, typename T>
IndexType
crout(IndexType n, T *A, IndexType cs)
{
    typedef decltype(std::real(T())) Real;

    for (IndexType j = 0; j < n; ++j) {
	T *c = &A[j * cs];
	IndexType k = 0;
	for (; k + 4 <= j; k += 4) {
	    const T *a0 = &A[(k + 0) * cs], *a1 = &A[(k + 1) * cs];
	    const T *a2 = &A[(k + 2) * cs], *a3 = &A[(k + 3) * cs];
	    const T b0 = conjugate(a0[j]), b1 = conjugate(a1[j]);
	    const T b2 = conjugate(a2[j]), b3 = conjugate(a3[j]);
	    for (IndexType i = j; i < n; ++i) {
		c[i] -= a0[i] * b0 + a1[i] * b1 + a2[i] * b2 + a3[i] * b3;
	    }
	}
	for (; k < j; ++k) {
	    const T *a0 = &A[k * cs];
	    const T b0 = conjugate(a0[j]);
	    for (IndexType i = j; i < n; ++i) {
		c[i] -= a0[i] * b0;
	    }
	}
	Real cjj = std::real(c[j]);
	if (!(cjj > 0)) { // also NaN
	    return j + 1;
	}
	cjj = std::sqrt(cjj);
	c[j] = cjj;
	const Real r = Real(1) / cjj;
	for (IndexType i = j + 1; i < n; ++i) {
	    c[i] *= r;
	}
    }
    return 0;
}

//
//  Row by row (Cholesky-Banachiewicz) for unit column stride:
//  L(i,j) = (A(i,j) - L(i,0:j) L(j,0:j)^H) / L(j,j), dot products over
//  contiguous rows with four partial sums.
//
template <typename IndexType, typename T>
IndexType
rows(IndexType n, T *A, IndexType rs)
{
    typedef decltype(std::real(T())) Real;

    for (IndexType i = 0; i < n; ++i) {
	T *li = &A[i * rs];
	for (IndexType j = 0; j <= i; ++j) {
	    const T *lj = &A[j * rs];
	    T s0 = T(0), s1 = T(0), s2 = T(0), s3 = T(0);
	    IndexType k = 0;
	    for (; k + 4 <= j; k += 4) {
		s0 += li[k] * conjugate(lj[k]);
		s1 += li[k + 1] * conjugate(lj[k + 1]);
		s2 += li[k + 2] * conjugate(lj[k + 2]);
		s3 += li[k + 3] * conjugate(lj[k + 3]);
	    }
	    for (; k < j; ++k) {
		s0 += li[k] * conjugate(lj[k]);
	    }
	    const T s = li[j] - ((s0 + s1) + (s2 + s3));
	    if (j < i) {
		li[j] = s / std::real(lj[j]);
	    } else {
		const Real d = std::real(s);
		if (!(d > 0)) { // also NaN
		    return i + 1;
		}
		li[i] = std::sqrt(d);
	    }
	}
    }
    return 0;
}

} // namespace potf2_impl

template <typename IndexType, typename T>
IndexType
potf2(IndexType n, T *A, IndexType incRowA, IndexType incColA)
{
    ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T);

    if (incRowA == 1) {
	return potf2_impl::crout(n, A, incColA);
    }
    if (incColA == 1) {
	return potf2_impl::rows(n, A, incRowA);
    }
    return potf2_impl::rightLooking(n, A, incRowA, incColA);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_POTF2_TCC
