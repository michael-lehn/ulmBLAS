#ifndef ULMBLAS_LEVEL1_IAMAX_TCC
#define ULMBLAS_LEVEL1_IAMAX_TCC 1

#include <cmath>
#include <limits>
#include <ulmblas/level1/iamax.h>

namespace ulmBLAS {

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
//  |re|/4 + |im|/4 once |re| + |im| overflows.
//
template <typename IndexType, typename VX>
IndexType
iamax(IndexType n, const std::complex<VX> *x, IndexType incX)
{
    if (n <= 0) {
	return -1;
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
