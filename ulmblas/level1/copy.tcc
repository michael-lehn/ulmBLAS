#ifndef ULMBLAS_LEVEL1_COPY_TCC
#define ULMBLAS_LEVEL1_COPY_TCC 1

#include <complex>
#include <type_traits>
#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1/copy.h>
#include <ulmblas/level1/kernel/simd.h>

namespace ulmBLAS {

//
//  Contiguous vectors: loops vectorized by the compiler (dcopy n = 1000 on
//  heim: 3.8 elements per ns with the strided loop), y from a 64-byte
//  boundary (ulmblas/level1/kernel/simd.h); complex vectors as real
//  numbers (with std::complex GCC 12 did not vectorize: zcopy 3.8
//  against 7.9 for OpenBLAS)
//
template <typename IndexType, typename VX, typename VY>
void
copyLoop(IndexType n, const VX *x, VY *y)
{
    for (IndexType i = 0; i < n; ++i) {
	y[i] = x[i];
    }
}

//  (loops from 0 with shifted pointers: with the main loop from p GCC 12
//  compiled a slower loop, dcopy n = 100 6.0 against 9.7 elements per ns)
template <typename IndexType, typename VX, typename VY>
void
copyContiguous(IndexType n, bool conjX, const VX *x, VY *y)
{
    if (!conjX) {
	const IndexType p = storePeel(n, y);
	copyLoop(p, x, y);
	copyLoop(n - p, x + p, y + p);
    } else {
	for (IndexType i = 0; i < n; ++i) {
	    y[i] = conjugate(x[i]);
	}
    }
}

template <typename IndexType, typename T>
void
copyConjLoop(IndexType n, const T *x, T *y)
{
    for (IndexType i = 0; i < n; ++i) {
	y[2 * i] = x[2 * i];
	y[2 * i + 1] = -x[2 * i + 1];
    }
}

template <typename IndexType, typename T>
void
copyContiguous(IndexType n, bool conjX, const std::complex<T> *x,
               std::complex<T> *y)
{
    const T *x_ = reinterpret_cast<const T *>(x);
    T *y_ = reinterpret_cast<T *>(y);

    if (!conjX) {
	copyContiguous(2 * n, false, x_, y_);
    } else {
	const IndexType p = storePeel(n, y);
	copyConjLoop(p, x_, y_);
	copyConjLoop(n - p, x_ + 2 * p, y_ + 2 * p);
    }
}

template <typename IndexType, typename VX, typename VY>
void
copy(IndexType n, bool conjX, const VX *x, IndexType incX, VY *y,
     IndexType incY)
{
    //  clang, real numbers: the strided loop of before, which clang versions
    //  for increment 1 itself (scopy increment 2, n = 100 on the M5: 3.9
    //  against 3.5 elements per ns with the extra branch)
    const bool contiguous =
        !(SimpleLoops::value && std::is_arithmetic<VX>::value);
    if (contiguous && incX == IndexType(1) && incY == IndexType(1)) {
	copyContiguous(n, conjX, x, y);
    } else if (!conjX) {
	for (IndexType i = 0; i < n; ++i) {
	    y[i * incY] = x[i * incX];
	}
    } else {
	for (IndexType i = 0; i < n; ++i) {
	    y[i * incY] = conjugate(x[i * incX]);
	}
    }
}

template <typename IndexType, typename VX, typename VY>
void
copy(IndexType n, const VX *x, IndexType incX, VY *y, IndexType incY)
{
    copy(n, false, x, incX, y, incY);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_COPY_TCC 1
