#ifndef ULMBLAS_LEVEL1_ROT_TCC
#define ULMBLAS_LEVEL1_ROT_TCC 1

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <ulmblas/level1/rot.h>

namespace ulmBLAS {

template <typename IndexType, typename VX, typename VY, typename T>
void
rot(IndexType n, VX *x, IndexType incX, VY *y, IndexType incY, T c, T s)
{
    for (IndexType i = 0; i < n; ++i) {
	VX x_ = c * x[i * incX] + s * y[i * incY];
	VY y_ = -s * x[i * incX] + c * y[i * incY];
	x[i * incX] = x_;
	y[i * incY] = y_;
    }
}

template <typename IndexType, typename X, typename Y, typename T>
void
rot(IndexType n, std::complex<X> *x, IndexType incX, std::complex<Y> *y,
    IndexType incY, T c, const std::complex<T> &s)
{
    typedef std::complex<T> CT;

    if (incX != IndexType(1) || incY != IndexType(1)) {
	for (IndexType i = 0; i < n; ++i) {
	    const CT tmp = c * x[i * incX] + s * y[i * incY];
	    y[i * incY] = c * y[i * incY] - std::conj(s) * x[i * incX];
	    x[i * incX] = tmp;
	}
    } else {
	for (IndexType i = 0; i < n; ++i) {
	    const CT tmp = c * x[i] + s * y[i];
	    y[i] = c * y[i] - std::conj(s) * x[i];
	    x[i] = tmp;
	}
    }
}

//
//  Smallest and largest safe scaling factors of the Reference BLAS (safmin,
//  safmax in DROTG, ZROTG of LAPACK 3.12.1)
//
template <typename T>
T
rotgSafmin()
{
    return std::numeric_limits<T>::min();
}

template <typename T>
T
rotgSafmax()
{
    return std::ldexp(T(1), std::numeric_limits<T>::max_exponent - 1);
}

//
//  Real Givens rotation as DROTG of LAPACK 3.12.1 (Anderson, "Algorithm 978:
//  Safe Scaling in the Level 1 BLAS"): r = sigma*sqrt(a^2 + b^2) with the
//  sign of the larger of a and b, scaled to avoid overflow and underflow.
//
template <typename A, typename B, typename T>
void
rotg(A &a, B &b, T &c, T &s)
{
    const T safmin = rotgSafmin<T>();
    const T safmax = rotgSafmax<T>();
    const T anorm = std::abs(a);
    const T bnorm = std::abs(b);

    if (bnorm == T(0)) {
	c = 1;
	s = 0;
	b = 0;
    } else if (anorm == T(0)) {
	c = 0;
	s = 1;
	a = b;
	b = 1;
    } else {
	const T scl =
	    std::min(safmax, std::max(safmin, std::max(anorm, bnorm)));
	const T sigma = std::copysign(T(1), anorm > bnorm ? T(a) : T(b));
	const T aScl = a / scl;
	const T bScl = b / scl;
	const T r = sigma * (scl * std::sqrt(aScl * aScl + bScl * bScl));

	c = a / r;
	s = b / r;

	T z;
	if (anorm > bnorm) {
	    z = s;
	} else if (c != T(0)) {
	    z = T(1) / c;
	} else {
	    z = 1;
	}
	a = r;
	b = z;
    }
}

//
//  Complex Givens rotation as ZROTG of LAPACK 3.12.1 (Anderson, "Algorithm
//  978"): c real, r = c*a + s*b and 0 = -conj(s)*a + c*b.  For b = 0: c = 1,
//  s = 0, r = a; for a = 0: c = 0, s = conj(b)/|b|, r = |b|; otherwise r has
//  the phase of a and |r| = sqrt(|a|^2 + |b|^2).  The operands are scaled if
//  |a|^2 + |b|^2 could overflow or underflow.  Products are computed as in
//  Fortran, without the special treatment of infinities of std::complex.
//
template <typename T>
std::complex<T>
rotgMul(const std::complex<T> &x, const std::complex<T> &y)
{
    return std::complex<T>(x.real() * y.real() - x.imag() * y.imag(),
                           x.real() * y.imag() + x.imag() * y.real());
}

template <typename T>
T
rotgAbsSq(const std::complex<T> &x)
{
    return x.real() * x.real() + x.imag() * x.imag();
}

template <typename TA, typename TB, typename T>
void
rotg(std::complex<TA> &a, std::complex<TB> &b, T &c, std::complex<T> &s)
{
    typedef std::complex<T> CT;

    const T safmin = rotgSafmin<T>();
    const T safmax = rotgSafmax<T>();
    const T rtmin = std::sqrt(safmin);
    const CT f = a, g = b;
    CT r;

    if (g == CT(0)) {
	c = 1;
	s = 0;
	r = f;
    } else if (f == CT(0)) {
	c = 0;
	if (g.real() == T(0)) {
	    r = std::abs(g.imag());
	    s = std::conj(g) / r.real();
	} else if (g.imag() == T(0)) {
	    r = std::abs(g.real());
	    s = std::conj(g) / r.real();
	} else {
	    const T g1 = std::max(std::abs(g.real()), std::abs(g.imag()));
	    const T rtmax = std::sqrt(safmax / 2);
	    if (g1 > rtmin && g1 < rtmax) {
		const T d = std::sqrt(rotgAbsSq(g));
		s = std::conj(g) / d;
		r = d;
	    } else {
		const T u = std::min(safmax, std::max(safmin, g1));
		const CT gs = g / u;
		const T d = std::sqrt(rotgAbsSq(gs));
		s = std::conj(gs) / d;
		r = d * u;
	    }
	}
    } else {
	const T f1 = std::max(std::abs(f.real()), std::abs(f.imag()));
	const T g1 = std::max(std::abs(g.real()), std::abs(g.imag()));
	T rtmax = std::sqrt(safmax / 4);

	if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
	    const T f2 = rotgAbsSq(f);
	    const T g2 = rotgAbsSq(g);
	    const T h2 = f2 + g2;
	    if (f2 >= h2 * safmin) {
		c = std::sqrt(f2 / h2);
		r = f / c;
		rtmax *= 2;
		if (f2 > rtmin && h2 < rtmax) {
		    s = rotgMul(std::conj(g), f / std::sqrt(f2 * h2));
		} else {
		    s = rotgMul(std::conj(g), r / h2);
		}
	    } else {
		const T d = std::sqrt(f2 * h2);
		c = f2 / d;
		if (c >= safmin) {
		    r = f / c;
		} else {
		    r = f * (h2 / d);
		}
		s = rotgMul(std::conj(g), f / d);
	    }
	} else {
	    const T u = std::min(safmax, std::max(safmin, std::max(f1, g1)));
	    const CT gs = g / u;
	    const T g2 = rotgAbsSq(gs);
	    T w, f2, h2;
	    CT fs;
	    if (f1 / u < rtmin) {
		const T v = std::min(safmax, std::max(safmin, f1));
		w = v / u;
		fs = f / v;
		f2 = rotgAbsSq(fs);
		h2 = f2 * w * w + g2;
	    } else {
		w = 1;
		fs = f / u;
		f2 = rotgAbsSq(fs);
		h2 = f2 + g2;
	    }
	    if (f2 >= h2 * safmin) {
		c = std::sqrt(f2 / h2);
		r = fs / c;
		rtmax *= 2;
		if (f2 > rtmin && h2 < rtmax) {
		    s = rotgMul(std::conj(gs), fs / std::sqrt(f2 * h2));
		} else {
		    s = rotgMul(std::conj(gs), r / h2);
		}
	    } else {
		const T d = std::sqrt(f2 * h2);
		c = f2 / d;
		if (c >= safmin) {
		    r = fs / c;
		} else {
		    r = fs * (h2 / d);
		}
		s = rotgMul(std::conj(gs), fs / d);
	    }
	    c *= w;
	    r *= u;
	}
    }
    a = r;
}

//
//  Modified Givens transformation: (x_i, y_i) = H*(x_i, y_i) with H given by
//  param[0] (flag) and param[1..4] = (h11, h21, h12, h22) as in the
//  Reference BLAS (SROTM, DROTM):
//
//      flag = -2:  H = I (x and y are not referenced)
//      flag = -1:  H = [h11 h12; h21 h22]
//      flag =  0:  H = [1 h12; h21 1]
//      flag =  1:  H = [h11 1; -1 h22]
//
//  Other values of flag are classified by their sign as in the Reference
//  BLAS.
//
template <typename IndexType, typename VX, typename VY, typename T>
void
rotm(IndexType n, VX *x, IndexType incX, VY *y, IndexType incY, const T *param)
{
    const T flag = param[0];

    if (n <= 0 || flag == T(-2)) {
	return;
    }

    T h11, h12, h21, h22;
    if (flag < T(0)) {
	h11 = param[1];
	h21 = param[2];
	h12 = param[3];
	h22 = param[4];
	for (IndexType i = 0; i < n; ++i) {
	    const VX w = x[i * incX];
	    const VY z = y[i * incY];
	    x[i * incX] = w * h11 + z * h12;
	    y[i * incY] = w * h21 + z * h22;
	}
    } else if (flag == T(0)) {
	h21 = param[2];
	h12 = param[3];
	for (IndexType i = 0; i < n; ++i) {
	    const VX w = x[i * incX];
	    const VY z = y[i * incY];
	    x[i * incX] = w + z * h12;
	    y[i * incY] = w * h21 + z;
	}
    } else {
	h11 = param[1];
	h22 = param[4];
	for (IndexType i = 0; i < n; ++i) {
	    const VX w = x[i * incX];
	    const VY z = y[i * incY];
	    x[i * incX] = w * h11 + z;
	    y[i * incY] = -w + h22 * z;
	}
    }
}

//
//  The scaling constants of ROTMG (gamma = 4096 and the bounds gamma^2 and
//  1/gamma^2 for d1 and |d2|), with the values of the Reference BLAS, where
//  the bounds of SROTMG and DROTMG are rounded decimal numbers.
//
template <typename T> struct RotmgScaling
{
    static T
    gam()
    {
	return T(4096);
    }
    static T
    gamSq()
    {
	return T(4096) * T(4096);
    }
    static T
    rgamSq()
    {
	return T(1) / (T(4096) * T(4096));
    }
};

template <> struct RotmgScaling<float>
{
    static float
    gam()
    {
	return 4096.0f;
    }
    static float
    gamSq()
    {
	return 1.67772e7f;
    }
    static float
    rgamSq()
    {
	return 5.96046e-8f;
    }
};

template <> struct RotmgScaling<double>
{
    static double
    gam()
    {
	return 4096.0;
    }
    static double
    gamSq()
    {
	return 16777216.0;
    }
    static double
    rgamSq()
    {
	return 5.9604645e-8;
    }
};

//
//  Construct the modified Givens transformation H (see rotm) that zeros the
//  second component of (sqrt(d1)*x1, sqrt(d2)*y1)^T: on return
//  H*(x1, y1)^T = (x1', 0)^T and d1, d2, x1 are overwritten by d1', d2', x1'
//  with d1'*x1'^2 = d1*x1^2 + d2*y1^2.  The algorithm of the Reference BLAS
//  (SROTMG, DROTMG): d1 and |d2| are kept in [1/gamma^2, gamma^2] by
//  rescaling, which turns flag 0 and 1 into the full form flag = -1.  As in
//  the current Reference BLAS, H is zeroed (flag = -1) if d1 < 0 or the
//  transformation does not exist (u <= 0 or q2 < 0), and an H that is
//  already in full form is not reset when the second scale factor is
//  adjusted.  Only the elements of param that the flag uses are set.
//
template <typename T>
void
rotmg(T &d1, T &d2, T &x1, const T &y1, T *param)
{
    const T Zero(0), One(1);
    const T gam = RotmgScaling<T>::gam();
    const T gamSq = RotmgScaling<T>::gamSq();
    const T rgamSq = RotmgScaling<T>::rgamSq();

    T flag = -One, h11 = Zero, h12 = Zero, h21 = Zero, h22 = Zero;
    bool zero = false;

    if (d1 < Zero) {
	zero = true;
    } else {
	const T p2 = d2 * y1;
	if (p2 == Zero) {
	    param[0] = T(-2);
	    return;
	}
	const T p1 = d1 * x1;
	const T q2 = p2 * y1;
	const T q1 = p1 * x1;

	if (std::abs(q1) > std::abs(q2)) {
	    h21 = -y1 / x1;
	    h12 = p2 / p1;
	    const T u = One - h12 * h21;
	    if (u > Zero) {
		flag = Zero;
		d1 /= u;
		d2 /= u;
		x1 *= u;
	    } else {
		zero = true;
	    }
	} else if (q2 < Zero) {
	    zero = true;
	} else {
	    flag = One;
	    h11 = p1 / p2;
	    h22 = x1 / y1;
	    const T u = One + h11 * h22;
	    const T tmp = d2 / u;
	    d2 = d1 / u;
	    d1 = tmp;
	    x1 = y1 * u;
	}
    }
    if (zero) {
	flag = -One;
	h11 = h12 = h21 = h22 = Zero;
	d1 = d2 = x1 = Zero;
    } else {
	if (d1 != Zero) {
	    while (d1 <= rgamSq || d1 >= gamSq) {
		if (flag == Zero) {
		    h11 = h22 = One;
		} else if (flag == One) {
		    h21 = -One;
		    h12 = One;
		}
		flag = -One;
		if (d1 <= rgamSq) {
		    d1 *= gam * gam;
		    x1 /= gam;
		    h11 /= gam;
		    h12 /= gam;
		} else {
		    d1 /= gam * gam;
		    x1 *= gam;
		    h11 *= gam;
		    h12 *= gam;
		}
	    }
	}
	if (d2 != Zero) {
	    while (std::abs(d2) <= rgamSq || std::abs(d2) >= gamSq) {
		if (flag == Zero) {
		    h11 = h22 = One;
		} else if (flag == One) {
		    h21 = -One;
		    h12 = One;
		}
		flag = -One;
		if (std::abs(d2) <= rgamSq) {
		    d2 *= gam * gam;
		    h21 /= gam;
		    h22 /= gam;
		} else {
		    d2 /= gam * gam;
		    h21 *= gam;
		    h22 *= gam;
		}
	    }
	}
    }
    if (flag < Zero) {
	param[1] = h11;
	param[2] = h21;
	param[3] = h12;
	param[4] = h22;
    } else if (flag == Zero) {
	param[2] = h21;
	param[3] = h12;
    } else {
	param[1] = h11;
	param[4] = h22;
    }
    param[0] = flag;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_ROT_TCC 1
