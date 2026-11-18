#ifndef ULMBLAS_LEVEL1_NRM2_TCC
#define ULMBLAS_LEVEL1_NRM2_TCC 1

#include <complex>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <ulmblas/level1/kernel/reduce.h>
#include <ulmblas/level1/nrm2.h>

namespace ulmBLAS {

template <typename IndexType, typename VX>
VX
nrm2(IndexType n, const VX *x, IndexType incX)
{
    VX result;

    nrm2(n, x, incX, result);
    return result;
}

template <typename IndexType, typename VX>
VX
nrm2(IndexType n, const std::complex<VX> *x, IndexType incX)
{
    VX result;

    nrm2(n, x, incX, result);
    return result;
}

//
//  Fast path for float and double (ulmblas/level1/kernel/reduce.h): one
//  pass computes the sum of squares (partial sums) and the largest
//  absolute value.  NaN propagates into the sum of squares, so the result
//  is NaN if an element is NaN, otherwise Inf if an element is infinite,
//  as for the scaled sum of squares below.
//
//  float: the squares and the sum in double (no overflow or underflow is
//  possible).  double: if the largest absolute value is in [2^-480, 2^480],
//  the unscaled sum can neither overflow nor lose accuracy by underflow
//  (n < 2^63); otherwise a second pass scales by 2^600 or 2^-600.
//  heim, elements per ns, n = 1000, before / now / OpenBLAS 0.3.34: dnrm2
//  0.78 / 10.2 / 3.8, snrm2 0.28 / 7.2 / 5.4, dznrm2 0.40 / 5.2 / 2.0.
//
template <typename T> struct Nrm2Lanes : std::false_type
{
    typedef T Acc;
};

template <> struct Nrm2Lanes<float> : std::true_type
{
    typedef double Acc;
};

template <> struct Nrm2Lanes<double> : std::true_type
{
    typedef double Acc;
};

//  2^480 as a product of exact factors (constant folded)
inline double
nrm2Big()
{
    const double t60 = double(std::uint64_t(1) << 60);
    const double t120 = t60 * t60;
    const double t240 = t120 * t120;
    return t240 * t240;
}

//  x: n real numbers (contiguous complex vectors as 2n) or complex numbers
template <typename R, typename IndexType, typename T>
R
nrm2Lanes(IndexType n, const T *x, IndexType incX)
{
    typedef typename Nrm2Lanes<R>::Acc Acc;

    Acc ssq, absMax;
    reduce::sumSquares(n, x, incX, Acc(1), ssq, absMax);
    if (std::isnan(ssq) || sizeof(Acc) > sizeof(R)) {
	return R(std::sqrt(ssq));
    }
    const Acc big = nrm2Big(), small = Acc(1) / big;
    if (absMax >= small && absMax <= big) {
	return R(std::sqrt(ssq));
    }
    if (absMax == Acc(0) || std::isinf(absMax)) {
	return R(absMax);
    }
    //  2^600 or 2^-600: the scaled absMax is in [2^-474, 2^424]
    const Acc t120 = Acc(std::uint64_t(1) << 60) * Acc(std::uint64_t(1) << 60);
    const Acc s = absMax < small ? big * t120 : small / t120;
    reduce::sumSquares(n, x, incX, s, ssq, absMax);
    return R(std::sqrt(ssq) / s);
}

//
//  Other types: the scaled sum of squares of the Reference BLAS (before
//  LAPACK 3.10).  A
//  component equal to the current scale adds exactly one, so that two
//  infinite components give Inf instead of Inf/Inf = NaN; NaN propagates.
//
template <typename IndexType, typename VX, typename Result>
void
nrm2(IndexType n, const VX *x, IndexType incX, Result &result)
{
    const Result Zero(0), One(1);

    if (n < 1) {
	result = Zero;
    } else if (n == 1) {
	result = std::abs(*x);
    } else if (Nrm2Lanes<Result>::value && std::is_same<VX, Result>::value) {
	result = nrm2Lanes<Result>(n, x, incX);
    } else {
	Result scale = 0;
	Result ssq = 1;

	for (IndexType i = 0; i < n; ++i) {
	    if (x[i * incX] != Zero) {
		Result absXi = std::abs(x[i * incX]);
		if (scale < absXi) {
		    ssq = One + ssq * std::pow(scale / absXi, 2);
		    scale = absXi;
		} else if (absXi == scale) {
		    ssq += One;
		} else {
		    ssq += std::pow(absXi / scale, 2);
		}
	    }
	}
	result = scale * sqrt(ssq);
    }
}

template <typename IndexType, typename VX, typename Result>
void
nrm2(IndexType n, const std::complex<VX> *x, IndexType incX, Result &result)
{
    const Result Zero(0), One(1);

    if (n < 1) {
	result = Zero;
    } else if (Nrm2Lanes<Result>::value && std::is_same<VX, Result>::value) {
	//  contiguous: the 2n real numbers
	result = incX == IndexType(1)
	             ? nrm2Lanes<Result>(2 * n, reinterpret_cast<const VX *>(x),
	                                 IndexType(1))
	             : nrm2Lanes<Result>(n, x, incX);
    } else {
	Result scale = 0;
	Result ssq = 1;

	for (IndexType i = 0; i < n; ++i) {
	    if (std::real(x[i * incX]) != Zero) {
		Result absXi = std::abs(std::real(x[i * incX]));
		if (scale < absXi) {
		    ssq = One + ssq * std::pow(scale / absXi, 2);
		    scale = absXi;
		} else if (absXi == scale) {
		    ssq += One;
		} else {
		    ssq += std::pow(absXi / scale, 2);
		}
	    }
	    if (imag(x[i * incX]) != Zero) {
		Result absXi = std::abs(std::imag(x[i * incX]));
		if (scale < absXi) {
		    ssq = One + ssq * std::pow(scale / absXi, 2);
		    scale = absXi;
		} else if (absXi == scale) {
		    ssq += One;
		} else {
		    ssq += std::pow(absXi / scale, 2);
		}
	    }
	}
	result = scale * sqrt(ssq);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_NRM2_TCC 1
