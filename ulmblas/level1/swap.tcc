#ifndef ULMBLAS_LEVEL1_SWAP_TCC
#define ULMBLAS_LEVEL1_SWAP_TCC 1

#include <complex>
#include <utility>
#include <ulmblas/level1/kernel/simd.h>
#include <ulmblas/level1/swap.h>

namespace ulmBLAS {

//
//  Contiguous vectors: vectorized by the compiler; complex vectors as real
//  numbers (with std::complex GCC 12 did not vectorize: cswap n = 1000 on
//  heim 2.0 against 4.0 elements per ns for OpenBLAS)
//
template <typename IndexType, typename VX, typename VY>
void
swapContiguous(IndexType n, VX *x, VY *y)
{
    for (IndexType i = 0; i < n; ++i) {
	std::swap(x[i], y[i]);
    }
}

//  real numbers: long vectors in blocks (ulmblas/level1/kernel/simd.h)
template <typename IndexType, typename T>
void
swapContiguous(IndexType n, T *x, T *y)
{
    const int B = InPlaceBlocks::size;
    IndexType i = 0;

    if (SimpleLoops::value) {
	//  clang: one loop from 0 (sswap n = 100 on the M5 11.6 against 10.9
	//  elements per ns with the loops for GCC)
	for (IndexType j = 0; j < n; ++j) {
	    std::swap(x[j], y[j]);
	}
	return;
    }
    for (const IndexType p = storePeel(n, x); i < p; ++i) {
	std::swap(x[i], y[i]);
    }
    if (n >= IndexType(InPlaceBlocks::minN)) {
	for (; i + B <= n; i += B) {
	    T xs[B], ys[B];
	    ULMBLAS_SIMD
	    for (int k = 0; k < B; ++k) {
		xs[k] = x[i + k];
		ys[k] = y[i + k];
	    }
	    ULMBLAS_SIMD
	    for (int k = 0; k < B; ++k) {
		x[i + k] = ys[k];
		y[i + k] = xs[k];
	    }
	}
    }
    for (; i < n; ++i) {
	std::swap(x[i], y[i]);
    }
}

template <typename IndexType, typename T>
void
swapContiguous(IndexType n, std::complex<T> *x, std::complex<T> *y)
{
    swapContiguous(2 * n, reinterpret_cast<T *>(x), reinterpret_cast<T *>(y));
}

template <typename IndexType, typename VX, typename VY>
void
swap(IndexType n, VX *x, IndexType incX, VY *y, IndexType incY)
{
    if (incX == IndexType(1) && incY == IndexType(1)) {
	swapContiguous(n, x, y);
    } else {
	for (IndexType i = 0; i < n; ++i) {
	    std::swap(x[i * incX], y[i * incY]);
	}
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_SWAP_TCC 1
