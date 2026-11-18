#ifndef ULMBLAS_LEVEL1_SCAL_TCC
#define ULMBLAS_LEVEL1_SCAL_TCC 1

#include <complex>
#include <ulmblas/level1/kernel/simd.h>
#include <ulmblas/level1/scal.h>

namespace ulmBLAS {

//
//  x_i *= alpha for a contiguous vector: from a 64-byte boundary of x
//  (ulmblas/level1/kernel/simd.h), in blocks of 32 elements (omp simd,
//  unrolled by the compiler).  heim, dscal, elements per ns: n = 1000 one
//  vectorized loop 12.9, in blocks 26.2 (OpenBLAS 24.9); the strided loop
//  of before 4.2.
//
template <typename IndexType, typename Alpha, typename VX>
void
scaleLoop(IndexType n, const Alpha &alpha, VX *x)
{
    IndexType i = 0;

#if !defined(ULMBLAS_LEVEL1_SIMPLE_LOOPS)
    const int B = 32;
    for (; i + B <= n; i += B) {
	ULMBLAS_SIMD
	for (int k = 0; k < B; ++k) {
	    x[i + k] *= alpha;
	}
    }
#endif
    for (; i < n; ++i) {
	x[i] *= alpha;
    }
}

template <typename IndexType, typename Alpha, typename VX>
void
scaleContiguous(IndexType n, const Alpha &alpha, VX *x)
{
    const IndexType p = storePeel(n, x);
    for (IndexType i = 0; i < p; ++i) {
	x[i] *= alpha;
    }
    scaleLoop(n - p, alpha, x + p);
}

//
//  x = alpha*x; for alpha = 0 the elements are set to zero (also NaN and
//  Inf), as required for beta = 0 in level 2 and 3.
//
template <typename IndexType, typename Alpha, typename VX>
void
scal(IndexType n, const Alpha &alpha, VX *x, IndexType incX)
{
    //  clang: the strided loops of before, which clang versions for increment
    //  1 itself (dscal increment 2, n = 100 on the M5: 4.0 against 3.6
    //  elements per ns with the extra branch)
    const bool contiguous = !SimpleLoops::value && incX == IndexType(1);
    if (contiguous && alpha != Alpha(1) && alpha != Alpha(0)) {
	scaleContiguous(n, alpha, x);
    } else if (contiguous && alpha == Alpha(0)) {
	for (IndexType i = 0; i < n; ++i) {
	    x[i] = Alpha(0);
	}
    } else if (alpha != Alpha(1) && alpha != Alpha(0)) {
	for (IndexType i = 0; i < n; ++i) {
	    x[i * incX] *= alpha;
	}
    } else if (alpha == Alpha(0)) {
	for (IndexType i = 0; i < n; ++i) {
	    x[i * incX] = Alpha(0);
	}
    }
}

//
//  x = alpha*x as ?SCAL of the reference BLAS: x is unchanged for alpha = 1,
//  otherwise every element is multiplied, also for alpha = 0 (NaN and Inf
//  give NaN).  Complex products are computed as in Fortran, without the
//  special treatment of infinities of std::complex.
//
template <typename Alpha, typename VX>
VX
scalBlasProduct(const Alpha &alpha, const VX &x)
{
    return alpha * x;
}

template <typename T>
std::complex<T>
scalBlasProduct(const std::complex<T> &alpha, const std::complex<T> &x)
{
    return std::complex<T>(alpha.real() * x.real() - alpha.imag() * x.imag(),
                           alpha.real() * x.imag() + alpha.imag() * x.real());
}

//
//  Contiguous vectors: vectorized by the compiler; complex vectors as real
//  numbers (zscal n = 1000 on heim with std::complex 3.8, OpenBLAS 4.7
//  elements per ns)
//
template <typename IndexType, typename Alpha, typename VX>
void
scalBlasContiguous(IndexType n, const Alpha &alpha, VX *x)
{
    for (IndexType i = 0; i < n; ++i) {
	x[i] = scalBlasProduct(alpha, x[i]);
    }
}

template <typename IndexType, typename T>
void
scalBlasContiguous(IndexType n, const T &alpha, T *x)
{
    scaleContiguous(n, alpha, x);
}

template <typename IndexType, typename T>
void
scalBlasContiguous(IndexType n, const T &alpha, std::complex<T> *x)
{
    scaleContiguous(2 * n, alpha, reinterpret_cast<T *>(x));
}

template <typename IndexType, typename T>
void
scalBlasContiguous(IndexType n, const std::complex<T> &alpha,
                   std::complex<T> *x)
{
    const T ar = alpha.real(), ai = alpha.imag();
    T *x_ = reinterpret_cast<T *>(x);

    IndexType i = 0;
    for (const IndexType p = storePeel(n, x); i < p; ++i) {
	const T xr = x_[2 * i], xi = x_[2 * i + 1];
	x_[2 * i] = ar * xr - ai * xi;
	x_[2 * i + 1] = ar * xi + ai * xr;
    }
    for (; i < n; ++i) {
	const T xr = x_[2 * i], xi = x_[2 * i + 1];
	x_[2 * i] = ar * xr - ai * xi;
	x_[2 * i + 1] = ar * xi + ai * xr;
    }
}

template <typename IndexType, typename Alpha, typename VX>
void
scalBlas(IndexType n, const Alpha &alpha, VX *x, IndexType incX)
{
    if (alpha == Alpha(1)) {
	return;
    }
    if (!SimpleLoops::value && incX == IndexType(1)) {
	scalBlasContiguous(n, alpha, x);
    } else {
	for (IndexType i = 0; i < n; ++i) {
	    x[i * incX] = scalBlasProduct(alpha, x[i * incX]);
	}
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_SCAL_TCC 1
