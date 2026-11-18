#ifndef ULMBLAS_LEVEL1_KERNEL_REF_AXPY_TCC
#define ULMBLAS_LEVEL1_KERNEL_REF_AXPY_TCC 1

#include <complex>
#include <type_traits>
#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1/kernel/ref/axpy.h>
#include <ulmblas/level1/kernel/simd.h>

namespace ulmBLAS {
namespace ref {

//
//  alpha*x; complex products as in the Reference BLAS (Fortran, without the
//  special treatment of infinities of std::complex, whose check for NaN
//  prevents vectorization: zaxpy n = 1000 on heim 0.4 elements per ns).
//
template <typename Alpha, typename VX>
VX
axpyProduct(const Alpha &alpha, const VX &x)
{
    return alpha * x;
}

template <typename T>
std::complex<T>
axpyProduct(const std::complex<T> &alpha, const std::complex<T> &x)
{
    return std::complex<T>(alpha.real() * x.real() - alpha.imag() * x.imag(),
                           alpha.real() * x.imag() + alpha.imag() * x.real());
}

//
//  Contiguous vectors, y from a 64-byte boundary (ulmblas/level1/kernel/
//  simd.h); complex vectors as real numbers (zaxpy n = 1000 on
//  heim with std::complex and the products above 3.8, OpenBLAS 5.6
//  elements per ns)
//
template <typename IndexType, typename Alpha, typename VX, typename VY>
void
axpyLoop(IndexType n, bool conjX, const Alpha &alpha, const VX *x, VY *y)
{
    if (!conjX) {
	for (IndexType i = 0; i < n; ++i) {
	    y[i] += axpyProduct(alpha, x[i]);
	}
    } else {
	for (IndexType i = 0; i < n; ++i) {
	    y[i] += axpyProduct(alpha, conjugate(x[i]));
	}
    }
}

//  the first elements up to a 64-byte boundary of y separately (loops from
//  0 with shifted pointers, as in copy)
template <typename IndexType, typename Alpha, typename VX, typename VY>
void
axpyContiguous(IndexType n, bool conjX, const Alpha &alpha, const VX *x, VY *y)
{
    const IndexType p = storePeel(n, y);
    axpyLoop(p, conjX, alpha, x, y);
    axpyLoop(n - p, conjX, alpha, x + p, y + p);
}

//  complex vectors as real numbers (two loops: as two calls of a loop
//  function GCC 12 was slower for n = 100, zaxpy 2.8 against 3.5 elements
//  per ns)
template <typename IndexType, typename T>
void
axpyContiguous(IndexType n, bool conjX, const std::complex<T> &alpha,
               const std::complex<T> *x, std::complex<T> *y)
{
    const T ar = alpha.real(), ai = alpha.imag();
    const T *x_ = reinterpret_cast<const T *>(x);
    T *y_ = reinterpret_cast<T *>(y);

    const IndexType p = storePeel(n, y);
    if (!conjX) {
	for (IndexType i = 0; i < p; ++i) {
	    const T xr = x_[2 * i], xi = x_[2 * i + 1];
	    y_[2 * i] += ar * xr - ai * xi;
	    y_[2 * i + 1] += ar * xi + ai * xr;
	}
	for (IndexType i = p; i < n; ++i) {
	    const T xr = x_[2 * i], xi = x_[2 * i + 1];
	    y_[2 * i] += ar * xr - ai * xi;
	    y_[2 * i + 1] += ar * xi + ai * xr;
	}
    } else {
	for (IndexType i = 0; i < p; ++i) {
	    const T xr = x_[2 * i], xi = -x_[2 * i + 1];
	    y_[2 * i] += ar * xr - ai * xi;
	    y_[2 * i + 1] += ar * xi + ai * xr;
	}
	for (IndexType i = p; i < n; ++i) {
	    const T xr = x_[2 * i], xi = -x_[2 * i + 1];
	    y_[2 * i] += ar * xr - ai * xi;
	    y_[2 * i + 1] += ar * xi + ai * xr;
	}
    }
}

//
//  clang, real numbers: the strided loop of before, which clang versions for
//  increment 1 itself (daxpy increment 2, n = 100 on the M5: 3.8 against 3.4
//  elements per ns with the extra branch)
//
template <typename VX>
bool
contiguous()
{
    return !(SimpleLoops::value && std::is_arithmetic<VX>::value);
}

//
//  y += alpha*x.  The loop for contiguous vectors is vectorized by the
//  compiler (heim, n = 1000, elements per ns: daxpy 9.8 with the SSE kernel
//  of before, 9.4 now, OpenBLAS 9.3; saxpy 3.1 with the strided loop of
//  before, 18.7 now, OpenBLAS 17.9).
//
template <typename IndexType, typename Alpha, typename VX, typename VY>
void
axpy(IndexType n, const Alpha &alpha, const VX *x, IndexType incX, VY *y,
     IndexType incY)
{
    if (contiguous<VX>() && incX == IndexType(1) && incY == IndexType(1)) {
	axpyContiguous(n, false, alpha, x, y);
    } else {
	for (IndexType i = 0; i < n; ++i) {
	    y[i * incY] += axpyProduct(alpha, x[i * incX]);
	}
    }
}

template <typename IndexType, typename Alpha, typename VX, typename VY>
void
acxpy(IndexType n, const Alpha &alpha, const VX *x, IndexType incX, VY *y,
      IndexType incY)
{
    if (contiguous<VX>() && incX == IndexType(1) && incY == IndexType(1)) {
	axpyContiguous(n, true, alpha, x, y);
    } else {
	for (IndexType i = 0; i < n; ++i) {
	    y[i * incY] += axpyProduct(alpha, conjugate(x[i * incX]));
	}
    }
}

} // namespace ref
} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_KERNEL_REF_AXPY_TCC 1
