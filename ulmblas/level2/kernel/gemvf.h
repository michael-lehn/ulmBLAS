#ifndef ULMBLAS_LEVEL2_KERNEL_GEMVF_H
#define ULMBLAS_LEVEL2_KERNEL_GEMVF_H 1

#include <complex>
#include <ulmblas/level1/kernel/simd.h>

//
//  Fused kernels of gemv for contiguous operands of one type, plain C++
//  vectorized by the compiler (omp simd, as the level 1 kernels in
//  ulmblas/level1/kernel/).  Complex numbers are processed as pairs of real
//  numbers (no std::complex arithmetic in the loops).
//
//  columns: y += sum_f a[f] A(:,f), F columns of A at a time (A column
//  major, BLAS N): each element of y is loaded and stored once for F
//  multiply-adds (OpenBLAS dgemv_n: 4 columns, BLIS axpyf: 8).
//
//  rows: r[f] = A(f,:) x for F rows of A (A row major, BLAS T) with L
//  partial sums ("lanes") per row: F L accumulators (8 vector registers
//  hide the latency of the multiply-adds); x is loaded once for F rows
//  (OpenBLAS dgemv_t: 4 rows with one vector each, BLIS dotxf: 8 rows
//  with one vector each).
//

namespace ulmBLAS {
namespace level2 {

//
//  y[i] += sum_f a[f] A[i + f ldA], i < m, real numbers
//
template <int F, typename IndexType, typename T>
void
columns(IndexType m, const T *a, const T *A, IndexType ldA, T *y)
{
    const T *Af[F];
    T af[F];
    for (int f = 0; f < F; ++f) {
	Af[f] = A + f * ldA;
	af[f] = a[f];
    }
    ULMBLAS_SIMD
    for (IndexType i = 0; i < m; ++i) {
	T s = y[i];
	for (int f = 0; f < F; ++f) {
	    s += af[f] * Af[f][i];
	}
	y[i] = s;
    }
}

//
//  complex numbers as pairs of real numbers: y, A real views (A: 2 ldA
//  real numbers between columns); a[f] = alpha x_f; Conj: conj(A).  Two
//  sums per element, (pr, pi) = sum_f re(a_f) A_f and (qr, qi) = sum_f
//  im(a_f) A_f (as OpenBLAS's zgemv_n: two multiply-adds per column, one
//  swap of q per element of y):
//
//      a A:        y += (pr - qi, pi + qr)
//      a conj(A):  y += (pr + qi, qr - pi)
//
template <int F, bool Conj, typename IndexType, typename T>
void
columnsComplex(IndexType m, const std::complex<T> *a, const T *A, IndexType ldA,
               T *y)
{
    const T *Af[F];
    T ar[F], ai[F];
    for (int f = 0; f < F; ++f) {
	Af[f] = A + 2 * f * ldA;
	ar[f] = a[f].real();
	ai[f] = a[f].imag();
    }
    ULMBLAS_SIMD
    for (IndexType i = 0; i < m; ++i) {
	T pr = 0, pi = 0, qr = 0, qi = 0;
	for (int f = 0; f < F; ++f) {
	    const T xr = Af[f][2 * i], xi = Af[f][2 * i + 1];
	    pr += ar[f] * xr;
	    pi += ar[f] * xi;
	    qr += ai[f] * xr;
	    qi += ai[f] * xi;
	}
	if (Conj) {
	    y[2 * i] += pr + qi;
	    y[2 * i + 1] += qr - pi;
	} else {
	    y[2 * i] += pr - qi;
	    y[2 * i + 1] += pi + qr;
	}
    }
}

//
//  acc[f][k] += acc[f][k + W] for W, W/2, ..., Min (constant bounds: GCC 12
//  does not unroll a loop over W, see ulmblas/level1/kernel/reduce.h)
//
template <int W, int Min, bool Go = (W >= Min)> struct Halve
{
    template <int F, int L, typename T>
    static void
    run(T (&acc)[F][L])
    {
	for (int f = 0; f < F; ++f) {
	    ULMBLAS_SIMD
	    for (int k = 0; k < W; ++k) {
		acc[f][k] += acc[f][k + W];
	    }
	}
	Halve<W / 2, Min>::run(acc);
    }
};

template <int W, int Min> struct Halve<W, Min, false>
{
    template <int F, int L, typename T>
    static void
    run(T (&acc)[F][L])
    {
    }
};

//
//  r[f] = sum_j A[f ldA + j] x[j], j < n, f < F; L lanes per row.  The lanes
//  start with the first block (not zeros, see ulmblas/level1/kernel/
//  reduce.h) and are added pairwise.
//
template <int F, int L, typename IndexType, typename T>
void
rows(IndexType n, const T *A, IndexType ldA, const T *x, T *r)
{
    const T *Af[F];
    for (int f = 0; f < F; ++f) {
	Af[f] = A + f * ldA;
	r[f] = T(0);
    }
    IndexType j = 0;
    if (n >= L) {
	T acc[F][L];
	for (int f = 0; f < F; ++f) {
	    ULMBLAS_SIMD
	    for (int k = 0; k < L; ++k) {
		acc[f][k] = Af[f][k] * x[k];
	    }
	}
	for (j = L; j + L <= n; j += L) {
	    for (int f = 0; f < F; ++f) {
		ULMBLAS_SIMD
		for (int k = 0; k < L; ++k) {
		    acc[f][k] += Af[f][j + k] * x[j + k];
		}
	    }
	}
	Halve<L / 2, 1>::run(acc);
	for (int f = 0; f < F; ++f) {
	    r[f] = acc[f][0];
	}
    }
    for (; j < n; ++j) {
	for (int f = 0; f < F; ++f) {
	    r[f] += Af[f][j] * x[j];
	}
    }
}

//
//  Complex: r[f] = sum_j A(f,j) x[j] (Conj: conj(A(f,j))), A, x, xs real
//  views (2 n real numbers per row, 2 ldA between rows); xs is x with the
//  real and imaginary parts swapped (x[k ^ 1] in the loop instead: GCC 12
//  used 16-byte vectors and kept accumulators on the stack).  Two sums per
//  lane: p = A x and q = A xs elementwise; the even lanes of p are the
//  products of the real parts, the odd ones those of the imaginary parts:
//
//      A x:        re = p_even - p_odd,  im = q_even + q_odd
//      conj(A) x:  re = p_even + p_odd,  im = q_even - q_odd
//
//  L (even) real numbers per lane block.
//
template <int F, int L, bool Conj, typename IndexType, typename T>
void
rowsComplex(IndexType n, const T *A, IndexType ldA, const T *x, const T *xs,
            std::complex<T> *r)
{
    const T *Af[F];
    T pe[F], po[F], qe[F], qo[F];
    for (int f = 0; f < F; ++f) {
	Af[f] = A + 2 * f * ldA;
	pe[f] = po[f] = qe[f] = qo[f] = T(0);
    }
    const IndexType n2 = 2 * n;
    IndexType j = 0;
    if (n2 >= L) {
	T p[F][L], q[F][L];
	for (int f = 0; f < F; ++f) {
	    ULMBLAS_SIMD
	    for (int k = 0; k < L; ++k) {
		p[f][k] = Af[f][k] * x[k];
		q[f][k] = Af[f][k] * xs[k];
	    }
	}
	for (j = L; j + L <= n2; j += L) {
	    for (int f = 0; f < F; ++f) {
		ULMBLAS_SIMD
		for (int k = 0; k < L; ++k) {
		    p[f][k] += Af[f][j + k] * x[j + k];
		    q[f][k] += Af[f][j + k] * xs[j + k];
		}
	    }
	}
	//  the even and odd lanes (loops of constant length, unrolled: with
	//  Halve as for real numbers GCC 12 stored the lanes on the stack and
	//  loaded 32 bytes from two 16-byte stores)
	for (int f = 0; f < F; ++f) {
	    for (int k = 0; k < L; k += 2) {
		pe[f] += p[f][k];
		po[f] += p[f][k + 1];
		qe[f] += q[f][k];
		qo[f] += q[f][k + 1];
	    }
	}
    }
    for (; j < n2; j += 2) {
	for (int f = 0; f < F; ++f) {
	    pe[f] += Af[f][j] * x[j];
	    po[f] += Af[f][j + 1] * x[j + 1];
	    qe[f] += Af[f][j] * xs[j];
	    qo[f] += Af[f][j + 1] * xs[j + 1];
	}
    }
    for (int f = 0; f < F; ++f) {
	r[f] = Conj ? std::complex<T>(pe[f] + po[f], qe[f] - qo[f])
	            : std::complex<T>(pe[f] - po[f], qe[f] + qo[f]);
    }
}

//
//  A[i + f ldA] += a[f] x[i], i < m, f < F (ger, A column major): x loaded
//  once for F columns (but see FuseGer in ulmblas/config/level2.h)
//
template <int F, typename IndexType, typename T>
void
rank1(IndexType m, const T *a, const T *x, T *A, IndexType ldA)
{
    T *Af[F];
    T af[F];
    for (int f = 0; f < F; ++f) {
	Af[f] = A + f * ldA;
	af[f] = a[f];
    }
    ULMBLAS_SIMD
    for (IndexType i = 0; i < m; ++i) {
	const T xi = x[i];
	for (int f = 0; f < F; ++f) {
	    Af[f][i] += af[f] * xi;
	}
    }
}

} // namespace level2
} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_KERNEL_GEMVF_H
