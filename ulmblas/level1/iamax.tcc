#ifndef ULMBLAS_LEVEL1_IAMAX_TCC
#define ULMBLAS_LEVEL1_IAMAX_TCC 1

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <ulmblas/level1/iamax.h>
#include <ulmblas/level1/kernel/reduce.h>

namespace ulmBLAS {

//
//  float, double and their complex types (n >= 1): blocks of 8 KB.  Per
//  block the maximum of |x_i| (complex: |re| + |im|) and a NaN test with
//  partial maxima and sums (ulmblas/level1/kernel/reduce.h); only if the
//  maximum is larger than that of the blocks before, the block (in L1) is
//  searched for its first element with this value, in chunks of 16 with a
//  vectorized test (a scalar search cost half of the time, idamax n = 1000
//  on heim: 2.4 elements per ns).  Real: a block with NaN is searched for
//  the first NaN.  Complex: returns -1 for a block with NaN or a value
//  with Inf or overflowing |re| + |im| (then the scalar algorithm below
//  decides).
//
template <bool Unit, typename R, typename IndexType, typename T>
IndexType
iamaxBlocks(IndexType n, const T *x, IndexType incX)
{
    const bool complex = !std::is_same<T, R>::value;
    const IndexType blockSize = 8192 / sizeof(T);
    const R hugeVal = std::numeric_limits<R>::max();
    const IndexType p = Unit ? reduce::peel(n, x, IndexType(1)) : IndexType(0);

    IndexType iAbsMaxX = 0;
    R absMaxX = -1;

    for (IndexType i0 = 0; i0 < n;) {
	const IndexType i1 =
	    std::min(n, i0 + blockSize + (i0 == 0 ? p : IndexType(0)));
	const IndexType m = i1 - i0;
	const T *xb = x + i0 * incX;
	const reduce::AbsMax<Unit, IndexType, T> op = {xb, incX};

	R max;
	const bool noNaN =
	    reduce::absMax<Unit>(m, xb, incX, i0 == 0 ? p : IndexType(0), max);
	if (complex && (!noNaN || max > hugeVal)) {
	    return -1;
	}
	if (!noNaN) {
	    for (IndexType j = 0; j < m; ++j) {
		if (std::isnan(op.abs(j))) {
		    return i0 + j;
		}
	    }
	}
	if (max > absMaxX) {
	    IndexType j = 0;
	    for (; j + 16 <= m; j += 16) {
		int hit = 0;
		ULMBLAS_PRAGMA(omp simd reduction(| : hit))
		for (int k = 0; k < 16; ++k) {
		    hit |= op.abs(j + k) == max;
		}
		if (hit) {
		    break;
		}
	    }
	    while (op.abs(j) != max) {
		++j;
	    }
	    iAbsMaxX = i0 + j;
	    absMaxX = max;
	}
	i0 = i1;
    }
    return iAbsMaxX;
}

template <typename R, typename IndexType, typename T>
IndexType
iamaxBlocks(IndexType n, const T *x, IndexType incX)
{
    return incX == IndexType(1) ? iamaxBlocks<true, R>(n, x, incX)
                                : iamaxBlocks<false, R>(n, x, incX);
}

template <typename T> struct IamaxBlocks : std::is_floating_point<T>
{
};

//
//  As i?amax of Reference-LAPACK after 3.12.1: the index of the first NaN,
//  otherwise of the first element with maximal absolute value.
//
template <typename IndexType, typename VX>
IndexType
iamax(IndexType n, const VX *x, IndexType incX)
{
    if (n <= 0) {
	return -1;
    }
    //  clang: the scalar loop for short or strided vectors (isamax on the
    //  M5, n = 100: 3.9 against 2.1 elements per ns in blocks; increment 2:
    //  4.1 against 3.1)
    if (IamaxBlocks<VX>::value &&
        !(SimpleLoops::value && (incX != IndexType(1) || n < IndexType(256)))) {
	return iamaxBlocks<VX>(n, x, incX);
    }

    IndexType iAbsMaxX = 0;
    VX absMaxX = std::abs(x[0]);

    if (std::isnan(absMaxX)) {
	return 0;
    }
    for (IndexType i = 1; i < n; ++i) {
	VX absX = std::abs(x[i * incX]);
	if (!(absX <= absMaxX)) {
	    iAbsMaxX = i;
	    absMaxX = absX;
	    if (std::isnan(absMaxX)) {
		return i;
	    }
	}
    }
    return iAbsMaxX;
}

//
//  Complex version, as i{c,z}amax of Reference-LAPACK after 3.12.1, with
//  |re| + |im| as absolute value: the index of the first element with NaN in
//  the real or imaginary part.  Otherwise the index of the first element with
//  Inf, if any.  Otherwise the first maximum of |re| + |im|, compared as
//  |re|/4 + |im|/4 once |re| + |im| overflows.  Without NaN, Inf and
//  overflow: the first maximum of |re| + |im| (iamaxBlocks).
//
template <typename IndexType, typename VX>
IndexType
iamax(IndexType n, const std::complex<VX> *x, IndexType incX)
{
    if (n <= 0) {
	return -1;
    }
    if (IamaxBlocks<VX>::value) {
	const IndexType i = iamaxBlocks<VX>(n, x, incX);
	if (i >= 0) {
	    return i;
	}
    }

    const VX hugeVal = std::numeric_limits<VX>::max();
    IndexType iAbsMaxX = 0;
    VX absMaxX = -1;
    bool scaled = false;

    for (IndexType i = 0; i < n; ++i) {
	VX re = std::abs(x[i * incX].real());
	VX im = std::abs(x[i * incX].imag());

	if (std::isnan(re) || std::isnan(im)) {
	    return i;
	}
	if (re > hugeVal || im > hugeVal) {
	    for (IndexType j = i + 1; j < n; ++j) {
		if (std::isnan(x[j * incX].real()) ||
		    std::isnan(x[j * incX].imag())) {
		    return j;
		}
	    }
	    return i;
	}
	if (!scaled) {
	    VX absX = re + im;
	    if (absX > hugeVal) {
		scaled = true;
		iAbsMaxX = i;
		absMaxX = VX(0.25) * re + VX(0.25) * im;
	    } else if (absX > absMaxX) {
		iAbsMaxX = i;
		absMaxX = absX;
	    }
	} else {
	    VX absX = VX(0.25) * re + VX(0.25) * im;
	    if (absX > absMaxX) {
		iAbsMaxX = i;
		absMaxX = absX;
	    }
	}
    }
    return iAbsMaxX;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_IAMAX_TCC 1
