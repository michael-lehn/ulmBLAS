#ifndef ULMBLAS_LEVEL1_KERNEL_REDUCE_H
#define ULMBLAS_LEVEL1_KERNEL_REDUCE_H 1

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <ulmblas/level1/kernel/simd.h>

//
//  Reductions over vectors (sums for asum, dot, sdsdot, nrm2; maxima for
//  nrm2 and iamax) with K partial results ("lanes") in the main loop: lane
//  k accumulates the terms of the elements i = p + k mod K.  K is 128 bytes
//  of accumulators (16 double, 32 float): 4 AVX2 or 8 NEON registers per
//  sum, enough to hide the latency of the additions.  The loop over the
//  lanes is vectorized by the compiler (plain C++ and omp simd, no
//  -ffast-math: the order of the operations is that of the source).
//
//  The first p elements up to a 64-byte boundary of x are peeled (loads
//  that cross a cache line halve the bandwidth from L2: dasum n = 100000
//  11.8 against 18.5 elements per ns on heim).  The peeled elements, the
//  lanes and the last elements (fewer than K) are then added in this
//  order.  So the order of the sum depends on n and on the alignment of x,
//  as in OpenBLAS.
//
//  The omp simd reduction of before gave GCC 12 one vector of partial sums
//  (latency bound).  heim (i5-12500, one core), elements per ns, n = 1000,
//  before / now / OpenBLAS 0.3.34: ddot 5.6 / 14.9 / 13.6, sdot 11.8 /
//  25.4 / 27.3, dasum 10.2 / 22.2 / 18.0, n = 100000: dasum 8.0 / 18.8 /
//  18.8.
//
//  Code generation with GCC 12, measured on heim: the loop over the lanes
//  is vectorized as a loop (omp simd; unrolled first, GCC vectorized only
//  part of the lanes: ddot n = 1000 8.4 instead of 14 elements per ns);
//  the lanes are initialized by the first block instead of zeros (the
//  zeros were stored in 16-byte pieces and loaded as 32 bytes, failed store
//  forwarding: 12 ns per call); all lanes are added alike to a separate
//  sum (with lane 0 as sum GCC split the lanes into groups of 1, 4, 4, 4,
//  2, 1); no pairwise tree in memory (20 ns).
//

namespace ulmBLAS {
namespace reduce {

template <typename Acc> struct Lanes
{
    enum
    {
	K = 128 / sizeof(Acc)
    };
};

//  Number of elements of x before the first one at a 64-byte boundary
//  (at most n), a multiple of 'multiple'
template <typename IndexType, typename T>
IndexType
peel(IndexType n, const T *x, IndexType multiple)
{
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(x);
    if (a % sizeof(T) != 0) {
	return 0;
    }
    IndexType p = IndexType((64 - a % 64) % 64 / sizeof(T));
    p -= p % multiple;
    return p < n ? p : n;
}

//
//  The lanes a[k], b[k] (k = 0, ..., Count-1) of the elements i + k with
//  op.one; Init: the lanes are not initialized yet (start from zero)
//
template <bool Init, int Count, typename Acc, typename IndexType, typename Op>
ULMBLAS_LEVEL1_INLINE void
block(const Op &op, Acc *a, Acc *b, IndexType i)
{
    ULMBLAS_SIMD
    for (int k = 0; k < Count; ++k) {
	Acc ak = Init ? Acc(0) : a[k];
	Acc bk = Init ? Acc(0) : b[k];
	op.one(ak, bk, i + k);
	a[k] = ak;
	b[k] = bk;
    }
}

//
//  Pairwise combination of the lanes k and k + W for k < W, then W/2, ...,
//  Groups (constant loop bounds: a loop over W is not unrolled by GCC 12)
//
template <int W, int Groups, bool Go = (W >= Groups)> struct Tree
{
    template <typename Acc, typename Op>
    static void
    run(const Op &op, Acc *a, Acc *b)
    {
	ULMBLAS_SIMD
	for (int k = 0; k < W; ++k) {
	    op.merge(a[k], b[k], a[k + W], b[k + W]);
	}
	Tree<W / 2, Groups>::run(op, a, b);
    }
};

template <int W, int Groups> struct Tree<W, Groups, false>
{
    template <typename Acc, typename Op>
    static void
    run(const Op &op, Acc *a, Acc *b)
    {
    }
};

//
//  Two reductions a, b over the elements 0, ..., n-1 in Groups groups (the
//  elements i = g mod Groups, Groups = 1 or 2) with
//
//      op.one(a_, b_, i):        add the terms of element i to a_, b_
//      op.block<Init, Count>(ra, rb, i):  the same for the elements i + k
//                                and the lanes ra[k], rb[k], k < Count
//      op.merge(a_, b_, a2, b2): combine two partial results
//
//  The results of group g are a[g], b[g].  A reduction that an operation
//  does not use is removed by the compiler.  The first p elements are
//  peeled (p even for Groups = 2).
//
template <int Groups, typename Acc, typename IndexType, typename Op>
void
lanes(const Op &op, IndexType n, IndexType p, Acc *a, Acc *b)
{
    const int K = Lanes<Acc>::K;
    Acc sa[Groups], sb[Groups];

    for (int g = 0; g < Groups; ++g) {
	sa[g] = sb[g] = Acc(0);
    }
    IndexType i = 0;
    for (; i < p; ++i) {
	op.one(sa[i % Groups], sb[i % Groups], i);
    }
    if (i + K <= n) {
	Acc ra[K], rb[K];

	op.template block<true, K>(ra, rb, i);
	for (i += K; i + K <= n; i += K) {
	    op.template block<false, K>(ra, rb, i);
	}
	//  the last elements in steps of K/2 and K/4 lanes (one chain of up to
	//  31 dependent additions cost 25 ns for float)
	if (i + K / 2 <= n) {
	    op.template block<false, K / 2>(ra, rb, i);
	    i += K / 2;
	}
	if (i + K / 4 <= n) {
	    op.template block<false, K / 4>(ra, rb, i);
	    i += K / 4;
	}
	Tree<K / 2, Groups>::run(op, ra, rb);
	for (int g = 0; g < Groups; ++g) {
	    op.merge(sa[g], sb[g], ra[g], rb[g]);
	}
    }
    for (; i < n; ++i) {
	op.one(sa[i % Groups], sb[i % Groups], i);
    }
    for (int g = 0; g < Groups; ++g) {
	a[g] = sa[g];
	b[g] = sb[g];
    }
}

//
//  Sum a and maximum b over the elements 0, ..., n-1 (op.one as in lanes):
//  Simple (clang, float elements, see ulmblas/level1/kernel/simd.h): one
//  omp simd loop, the compiler chooses the partial sums; otherwise lanes
//  with p peeled elements.
//
template <bool Simple, bool Short, typename Acc, typename IndexType,
          typename Op>
void
sumMax(const Op &op, IndexType n, IndexType p, Acc &a, Acc &b)
{
    //  Short: short vectors in one loop as well (clang: ddot n = 100 on the
    //  M5 7.9 in lanes, 10.0 in one loop)
    if (Simple || (Short && n < IndexType(256))) {
	Acc sa = Acc(0), sb = Acc(0);
	ULMBLAS_PRAGMA(omp simd reduction(+ : sa) reduction(max : sb))
	for (IndexType i = 0; i < n; ++i) {
	    op.one(sa, sb, i);
	}
	a = sa;
	b = sb;
    } else {
	Acc ra[1], rb[1];
	lanes<1>(op, n, p, ra, rb);
	a = ra[0];
	b = rb[0];
    }
}

//
//  Element i of x with increment inc; Unit: inc = 1 (contiguous, known to
//  the compiler)
//
template <bool Unit, typename IndexType, typename T>
const T &
at(const T *x, IndexType inc, IndexType i)
{
    return x[Unit ? i : i * inc];
}

//
//  asum: a += |x_i|; complex: |re| + |im| (contiguous complex vectors are
//  summed as 2n real numbers)
//
template <typename Acc, bool Unit, typename IndexType, typename T> struct SumAbs
{
    const T *x;
    IndexType inc;

    Acc
    term(IndexType i) const
    {
	return Acc(std::abs(at<Unit>(x, inc, i)));
    }

    void
    one(Acc &a, Acc &unused, IndexType i) const
    {
	a += term(i);
    }

    void
    merge(Acc &a, Acc &unused, Acc a2, Acc unused2) const
    {
	a += a2;
    }

    template <bool Init, int Count>
    ULMBLAS_LEVEL1_INLINE void
    block(Acc *a, Acc *b, IndexType i) const
    {
	reduce::block<Init, Count>(*this, a, b, i);
    }
};

template <typename Acc, bool Unit, typename IndexType, typename T>
struct SumAbs<Acc, Unit, IndexType, std::complex<T>>
{
    const std::complex<T> *x;
    IndexType inc;

    void
    one(Acc &a, Acc &unused, IndexType i) const
    {
	const std::complex<T> &z = at<Unit>(x, inc, i);
	a += Acc(std::abs(z.real())) + Acc(std::abs(z.imag()));
    }

    void
    merge(Acc &a, Acc &unused, Acc a2, Acc unused2) const
    {
	a += a2;
    }

    template <bool Init, int Count>
    ULMBLAS_LEVEL1_INLINE void
    block(Acc *a, Acc *b, IndexType i) const
    {
	reduce::block<Init, Count>(*this, a, b, i);
    }
};

//  |x_0| + ... + |x_{n-1}| in the precision of Acc
template <typename Acc, typename IndexType, typename T>
Acc
sumAbs(IndexType n, const T *x, IndexType incX)
{
    Acc a, b;

    if (incX == IndexType(1)) {
	SumAbs<Acc, true, IndexType, T> op = {x, incX};
	sumMax<SimpleSum<T>::value, SimpleLoops::value>(
	    op, n, peel(n, x, IndexType(1)), a, b);
    } else {
	SumAbs<Acc, false, IndexType, T> op = {x, incX};
	sumMax<SimpleSum<T>::value, SimpleLoops::value>(op, n, IndexType(0), a,
	                                                b);
    }
    return a;
}

//
//  dot: a += x_i y_i, products and sum in the precision of Acc
//
template <typename Acc, bool Unit, typename IndexType, typename TX, typename TY>
struct Dot
{
    const TX *x;
    IndexType incX;
    const TY *y;
    IndexType incY;

    void
    one(Acc &a, Acc &unused, IndexType i) const
    {
	a += Acc(at<Unit>(x, incX, i)) * Acc(at<Unit>(y, incY, i));
    }

    void
    merge(Acc &a, Acc &unused, Acc a2, Acc unused2) const
    {
	a += a2;
    }

    template <bool Init, int Count>
    ULMBLAS_LEVEL1_INLINE void
    block(Acc *a, Acc *b, IndexType i) const
    {
	reduce::block<Init, Count>(*this, a, b, i);
    }
};

template <typename Acc, typename IndexType, typename TX, typename TY>
Acc
dot(IndexType n, const TX *x, IndexType incX, const TY *y, IndexType incY)
{
    Acc a, b;

    if (incX == IndexType(1) && incY == IndexType(1)) {
	//  peeled only for vectors in the caches: from memory the aligned loads
	//  were slower, ddot n = 10^7 on heim 1.27 against 1.40 elements per
	//  ns (OpenBLAS 1.45), but for n = 100000 sdot 18.5 against 13.3
	Dot<Acc, true, IndexType, TX, TY> op = {x, incX, y, incY};
	const bool cached = n <= IndexType((std::size_t(1) << 21) / sizeof(TX));
	sumMax<SimpleSum<TX>::value, SimpleLoops::value>(
	    op, n, cached ? peel(n, x, IndexType(1)) : IndexType(0), a, b);
    } else {
	Dot<Acc, false, IndexType, TX, TY> op = {x, incX, y, incY};
	sumMax<SimpleSum<TX>::value, SimpleLoops::value>(op, n, IndexType(0), a,
	                                                 b);
    }
    return a;
}

//
//  Complex dot op(x)^T y with op(x) = x or conj(x), products as in the
//  Reference BLAS (Fortran, without the treatment of infinities of
//  std::complex).
//
//  Contiguous: x and y as 2n real numbers, a_j += x_j y_j and b_j += x_j
//  y_{j^1} (j^1: the other part of the same element) in two groups (j even
//  and odd); then for x^T y: re = sum of a over even j - sum over odd j,
//  im = sum of b; for x^H y: re = sum of a, im = even - odd sum of b.
//
//  The products of a block are computed in a fully unrolled loop, which
//  GCC 12 vectorizes with one permutation (vpermilpd) per vector, and then
//  added in a vectorized loop.  (One loop: GCC deinterleaved x and y, or
//  with loads shifted by one element instead of the permutation the loads
//  were the limit; in lanes() GCC kept the lanes in memory.  zdotc n = 1000
//  on heim: 2.3 elements per ns, now 4.5, OpenBLAS 6.3.)  The peeled and
//  the last elements are added to separate sums for even and odd j.
//
template <typename IndexType, typename T>
std::complex<T>
dotComplex(IndexType n, bool conjX, const T *x, const T *y)
{
    const int K = Lanes<T>::K;
    const IndexType m = 2 * n;
    const IndexType p = peel(m, x, IndexType(2));
    T a[K] = {}, b[K] = {};
    T s[4] = {}; //  even and odd sums of a, b

    IndexType j = 0;
    for (; j < p; ++j) {
	s[j % 2] += x[j] * y[j];
	s[2 + j % 2] += x[j] * y[j ^ 1];
    }
    for (; j + K <= m; j += K) {
	T ta[K], tb[K];
	for (int k = 0; k < K; k += 2) {
	    ta[k] = x[j + k] * y[j + k];
	    ta[k + 1] = x[j + k + 1] * y[j + k + 1];
	    tb[k] = x[j + k] * y[j + k + 1];
	    tb[k + 1] = x[j + k + 1] * y[j + k];
	}
	ULMBLAS_SIMD
	for (int k = 0; k < K; ++k) {
	    a[k] += ta[k];
	    b[k] += tb[k];
	}
    }
    for (int k = 0; k < K; ++k) {
	s[k % 2] += a[k];
	s[2 + k % 2] += b[k];
    }
    for (; j < m; ++j) {
	s[j % 2] += x[j] * y[j];
	s[2 + j % 2] += x[j] * y[j ^ 1];
    }
    return conjX ? std::complex<T>(s[0] + s[1], s[2] - s[3])
                 : std::complex<T>(s[0] - s[1], s[2] + s[3]);
}

//  Strided: per element re += xr yr - s xi yi, im += xr yi + s xi yr
//  (s = -1 for conj(x))
template <typename IndexType, typename T> struct DotComplexStrided
{
    const std::complex<T> *x;
    IndexType incX;
    const std::complex<T> *y;
    IndexType incY;
    T s;

    void
    one(T &re, T &im, IndexType i) const
    {
	const T xr = x[i * incX].real(), xi = s * x[i * incX].imag();
	const T yr = y[i * incY].real(), yi = y[i * incY].imag();
	re += xr * yr - xi * yi;
	im += xr * yi + xi * yr;
    }

    void
    merge(T &a, T &b, T a2, T b2) const
    {
	a += a2;
	b += b2;
    }

    template <bool Init, int Count>
    ULMBLAS_LEVEL1_INLINE void
    block(T *a, T *b, IndexType i) const
    {
	reduce::block<Init, Count>(*this, a, b, i);
    }
};

template <typename IndexType, typename T>
std::complex<T>
dot(IndexType n, bool conjX, const std::complex<T> *x, IndexType incX,
    const std::complex<T> *y, IndexType incY)
{
    T a[2], b[2];

    if (incX == IndexType(1) && incY == IndexType(1) &&
        (SimpleSum<T>::value || (SimpleLoops::value && n < IndexType(256)))) {
	//  one omp simd loop (clang, float: cdotc n = 1000 on the M5 6.0
	//  against 5.7 elements per ns with dotComplex; double, n = 100: zdotc
	//  2.8 against 2.6)
	const T *x_ = reinterpret_cast<const T *>(x);
	const T *y_ = reinterpret_cast<const T *>(y);
	const T s = conjX ? T(-1) : T(1);
	T re = T(0), im = T(0);
	ULMBLAS_PRAGMA(omp simd reduction(+ : re, im))
	for (IndexType i = 0; i < n; ++i) {
	    const T xr = x_[2 * i], xi = s * x_[2 * i + 1];
	    const T yr = y_[2 * i], yi = y_[2 * i + 1];
	    re += xr * yr - xi * yi;
	    im += xr * yi + xi * yr;
	}
	return std::complex<T>(re, im);
    }
    if (incX == IndexType(1) && incY == IndexType(1)) {
	return dotComplex(n, conjX, reinterpret_cast<const T *>(x),
	                  reinterpret_cast<const T *>(y));
    }
    DotComplexStrided<IndexType, T> op = {x, incX, y, incY,
                                          conjX ? T(-1) : T(1)};
    if (SimpleLoops::value) {
	//  clang: one loop (cdotc increment 2, n = 100 on the M5: 1.7 against
	//  1.5 elements per ns in lanes)
	T re = T(0), im = T(0);
	for (IndexType i = 0; i < n; ++i) {
	    op.one(re, im, i);
	}
	return std::complex<T>(re, im);
    }
    lanes<1>(op, n, IndexType(0), a, b);
    return std::complex<T>(a[0], b[0]);
}

//
//  nrm2: a += (s x_i)^2 and b = max(b, |x_i|) in the precision of Acc;
//  complex: both parts.  NaN propagates into a, but not into b.
//
template <typename Acc, bool Unit, typename IndexType, typename T>
struct SumSquares
{
    const T *x;
    IndexType inc;
    Acc s;

    void
    one(Acc &a, Acc &b, IndexType i) const
    {
	const Acc v = Acc(at<Unit>(x, inc, i));
	const Acc w = s * v;
	const Acc abs = std::abs(v);
	a += w * w;
	b = abs > b ? abs : b;
    }

    void
    merge(Acc &a, Acc &b, Acc a2, Acc b2) const
    {
	a += a2;
	b = b2 > b ? b2 : b;
    }

    template <bool Init, int Count>
    ULMBLAS_LEVEL1_INLINE void
    block(Acc *a, Acc *b, IndexType i) const
    {
	reduce::block<Init, Count>(*this, a, b, i);
    }
};

template <typename Acc, bool Unit, typename IndexType, typename T>
struct SumSquares<Acc, Unit, IndexType, std::complex<T>>
{
    const std::complex<T> *x;
    IndexType inc;
    Acc s;

    void
    one(Acc &a, Acc &b, IndexType i) const
    {
	const std::complex<T> &z = at<Unit>(x, inc, i);
	const Acc re = s * Acc(z.real()), im = s * Acc(z.imag());
	const Acc absRe = std::abs(Acc(z.real()));
	const Acc absIm = std::abs(Acc(z.imag()));
	const Acc abs = absRe > absIm ? absRe : absIm;
	a += re * re + im * im;
	b = abs > b ? abs : b;
    }

    void
    merge(Acc &a, Acc &b, Acc a2, Acc b2) const
    {
	a += a2;
	b = b2 > b ? b2 : b;
    }

    template <bool Init, int Count>
    ULMBLAS_LEVEL1_INLINE void
    block(Acc *a, Acc *b, IndexType i) const
    {
	reduce::block<Init, Count>(*this, a, b, i);
    }
};

//  ssq = sum of (s x_i)^2, absMax = max |x_i| (real and imaginary parts)
template <typename Acc, typename IndexType, typename T>
void
sumSquares(IndexType n, const T *x, IndexType incX, Acc s, Acc &ssq,
           Acc &absMax)
{
    if (incX == IndexType(1)) {
	SumSquares<Acc, true, IndexType, T> op = {x, incX, s};
	sumMax<SimpleSum<T>::value, false>(op, n, peel(n, x, IndexType(1)), ssq,
	                                   absMax);
    } else {
	SumSquares<Acc, false, IndexType, T> op = {x, incX, s};
	//  strided: lanes (clang does not vectorize the one loop with the
	//  maximum)
	sumMax<false, false>(op, n, IndexType(0), ssq, absMax);
    }
}

//
//  iamax: a = max(a, |x_i|), b += |x_i|; complex: |re| + |im|.  The sum b
//  is NaN if and only if an element is NaN (the terms are not negative,
//  Inf + Inf = Inf).
//
template <bool Unit, typename IndexType, typename T> struct AbsMax
{
    const T *x;
    IndexType inc;

    T
    abs(IndexType i) const
    {
	return std::abs(at<Unit>(x, inc, i));
    }

    void
    one(T &a, T &b, IndexType i) const
    {
	const T v = abs(i);
	a = v > a ? v : a;
	b += v;
    }

    void
    merge(T &a, T &b, T a2, T b2) const
    {
	a = a2 > a ? a2 : a;
	b += b2;
    }

    template <bool Init, int Count>
    ULMBLAS_LEVEL1_INLINE void
    block(T *a, T *b, IndexType i) const
    {
	reduce::block<Init, Count>(*this, a, b, i);
    }
};

template <bool Unit, typename IndexType, typename T>
struct AbsMax<Unit, IndexType, std::complex<T>>
{
    const std::complex<T> *x;
    IndexType inc;

    T
    abs(IndexType i) const
    {
	const std::complex<T> &z = at<Unit>(x, inc, i);
	return std::abs(z.real()) + std::abs(z.imag());
    }

    void
    one(T &a, T &b, IndexType i) const
    {
	const T v = abs(i);
	a = v > a ? v : a;
	b += v;
    }

    void
    merge(T &a, T &b, T a2, T b2) const
    {
	a = a2 > a ? a2 : a;
	b += b2;
    }

    template <bool Init, int Count>
    ULMBLAS_LEVEL1_INLINE void
    block(T *a, T *b, IndexType i) const
    {
	reduce::block<Init, Count>(*this, a, b, i);
    }
};

//
//  Maximum of |x_i| (complex: |re| + |im|) over i = 0, ..., n-1 and
//  whether no element is NaN; p elements are peeled
//
template <bool Unit, typename IndexType, typename T, typename R>
bool
absMax(IndexType n, const T *x, IndexType incX, IndexType p, R &max)
{
    AbsMax<Unit, IndexType, T> op = {x, incX};

    R a[1], b[1];
    lanes<1>(op, n, p, a, b);
    max = a[0];
    return !std::isnan(b[0]);
}

} // namespace reduce
} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_KERNEL_REDUCE_H
