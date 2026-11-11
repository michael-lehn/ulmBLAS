//
//  GEMM with arbitrary strides (ulmblas_?gemm_strided, ulmblas_strided.h).
//  Also sbgemm and shgemm (A and B in bfloat16 or half precision, C in
//  float).
//
//  Every operand lives in a buffer of its own and is described by a
//  pointer to element (0,0) and two strides (column major, row major,
//  non-unit, negative, mixed, zero).  The result is compared with a
//  reference in complex<double>; the elements of the buffer of C that are
//  not elements of C must not change (bitwise).
//
#include <cstdint>
#include <type_traits>
#include <vector>
#include <ulmblas_strided.h>
#include <ulmblas/auxiliary/bfloat16.h>
#include <ulmblas/auxiliary/float16.h>
#include <ulmblas/ulmblas.h>
#include <test/numeric/reference.h>

using namespace test;

namespace {

//
//  The C functions for each type
//
void
gemmStrided(std::int64_t m, std::int64_t n, std::int64_t k, float alpha, bool,
            const float *A, std::int64_t rA, std::int64_t cA, bool,
            const float *B, std::int64_t rB, std::int64_t cB, float beta,
            float *C, std::int64_t rC, std::int64_t cC)
{
    ulmblas_sgemm_strided(m, n, k, alpha, A, rA, cA, B, rB, cB, beta, C, rC,
                          cC);
}

void
gemmStrided(std::int64_t m, std::int64_t n, std::int64_t k, double alpha, bool,
            const double *A, std::int64_t rA, std::int64_t cA, bool,
            const double *B, std::int64_t rB, std::int64_t cB, double beta,
            double *C, std::int64_t rC, std::int64_t cC)
{
    ulmblas_dgemm_strided(m, n, k, alpha, A, rA, cA, B, rB, cB, beta, C, rC,
                          cC);
}

void
gemmStrided(std::int64_t m, std::int64_t n, std::int64_t k, const cfloat &alpha,
            bool conjA, const cfloat *A, std::int64_t rA, std::int64_t cA,
            bool conjB, const cfloat *B, std::int64_t rB, std::int64_t cB,
            const cfloat &beta, cfloat *C, std::int64_t rC, std::int64_t cC)
{
    ulmblas_cgemm_strided(m, n, k, reinterpret_cast<const float *>(&alpha),
                          conjA, reinterpret_cast<const float *>(A), rA, cA,
                          conjB, reinterpret_cast<const float *>(B), rB, cB,
                          reinterpret_cast<const float *>(&beta),
                          reinterpret_cast<float *>(C), rC, cC);
}

void
gemmStrided(std::int64_t m, std::int64_t n, std::int64_t k,
            const cdouble &alpha, bool conjA, const cdouble *A, std::int64_t rA,
            std::int64_t cA, bool conjB, const cdouble *B, std::int64_t rB,
            std::int64_t cB, const cdouble &beta, cdouble *C, std::int64_t rC,
            std::int64_t cC)
{
    ulmblas_zgemm_strided(m, n, k, reinterpret_cast<const double *>(&alpha),
                          conjA, reinterpret_cast<const double *>(A), rA, cA,
                          conjB, reinterpret_cast<const double *>(B), rB, cB,
                          reinterpret_cast<const double *>(&beta),
                          reinterpret_cast<double *>(C), rC, cC);
}

//
//  Layouts of an r x c operand.  ZeroRow and ZeroCol (stride 0) only for
//  A and B, or for C with r = 1 (c = 1).
//
enum Layout
{
    ColMajor,
    RowMajor,
    RowStride2,
    NegRow,
    NegBoth,
    Mixed,
    ZeroRow,
    ZeroCol,
    NumLayouts
};

const char *
layoutName(Layout l)
{
    static const char *name[] = {"col",     "row",   "rowstride2", "negrow",
                                 "negboth", "mixed", "zerorow",    "zerocol"};
    return name[l];
}

template <typename T> struct Operand
{
    Operand(Layout layout, std::int64_t r_, std::int64_t c_) : r(r_), c(c_)
    {
	std::int64_t R = std::max<std::int64_t>(r, 1);
	std::int64_t C = std::max<std::int64_t>(c, 1);
	switch (layout) {
	case ColMajor:
	    rs = 1;
	    cs = R + 3;
	    break;
	case RowMajor:
	    rs = C + 2;
	    cs = 1;
	    break;
	case RowStride2:
	    rs = 2;
	    cs = 2 * R + 1;
	    break;
	case NegRow:
	    rs = -1;
	    cs = R + 1;
	    break;
	case NegBoth:
	    rs = -C;
	    cs = -1;
	    break;
	case Mixed:
	    rs = 3;
	    cs = -(3 * R + 2);
	    break;
	case ZeroRow:
	    rs = 0;
	    cs = 2;
	    break;
	default:
	    rs = 3;
	    cs = 0;
	    break;
	}
	//  elements at offsets i*rs + j*cs, 0 <= i < r, 0 <= j < c
	std::int64_t lo = 0, hi = 0;
	if (r > 0 && c > 0) {
	    lo = std::min<std::int64_t>(0, rs * (r - 1)) +
	         std::min<std::int64_t>(0, cs * (c - 1));
	    hi = std::max<std::int64_t>(0, rs * (r - 1)) +
	         std::max<std::int64_t>(0, cs * (c - 1));
	}
	//  a margin on both sides, to detect writes outside the matrix
	base = 5 - lo;
	buf.resize(std::size_t(hi - lo + 11));
    }

    T *
    ptr()
    {
	return &buf[base];
    }
    T &
    operator()(std::int64_t i, std::int64_t j)
    {
	return buf[base + i * rs + j * cs];
    }

    std::int64_t r, c, rs, cs, base;
    std::vector<T> buf;
};

template <typename T>
void
randomize(Random &random, std::vector<T> &v)
{
    for (auto &x : v) {
	x = random.value<T>();
    }
}

template <typename T>
T
conjIf(bool c, const T &x)
{
    return c ? conjugate(x) : x;
}

//
//  One case: C = alpha op(A) op(B) + beta C, A m x k, B k x n, C m x n
//
template <typename T>
bool
stridedCase(Context &ctx, Random &random, std::int64_t m, std::int64_t n,
            std::int64_t k, Layout lA, Layout lB, Layout lC, bool conjA,
            bool conjB, T alpha, T beta)
{
    Operand<T> A(lA, m, k), B(lB, k, n), C(lC, m, n);

    randomize(random, A.buf);
    randomize(random, B.buf);
    randomize(random, C.buf);
    if (alpha == T(0)) {
	//  A and B must not be read
	for (auto &x : A.buf)
	    x = nan<T>();
	for (auto &x : B.buf)
	    x = nan<T>();
    }
    if (beta == T(0)) {
	//  C must not be read
	for (std::int64_t j = 0; j < n; ++j) {
	    for (std::int64_t i = 0; i < m; ++i) {
		C(i, j) = nan<T>();
	    }
	}
    }
    std::vector<T> C0 = C.buf;

    //  reference and error bound
    std::vector<cdouble> ref(std::size_t(std::max<std::int64_t>(m * n, 1)));
    std::vector<double> bound(ref.size());
    for (std::int64_t j = 0; j < n; ++j) {
	for (std::int64_t i = 0; i < m; ++i) {
	    cdouble s = 0;
	    double b = 0;
	    if (alpha != T(0)) {
		for (std::int64_t l = 0; l < k; ++l) {
		    cdouble p = toCdouble(conjIf(conjA, A(i, l))) *
		                toCdouble(conjIf(conjB, B(l, j)));
		    s += p;
		    b += std::abs(p);
		}
	    }
	    cdouble a = toCdouble(alpha), be = toCdouble(beta);
	    cdouble c = beta == T(0) ? cdouble(0) : toCdouble(C(i, j));
	    ref[i + j * m] = a * s + be * c;
	    bound[i + j * m] = std::abs(a) * b + std::abs(be) * std::abs(c);
	}
    }

    gemmStrided(m, n, k, alpha, conjA, A.ptr(), A.rs, A.cs, conjB, B.ptr(),
                B.rs, B.cs, beta, C.ptr(), C.rs, C.cs);

    std::string what =
        format("gemm_strided<%s>(m=%d,n=%d,k=%d,A %s%s,"
	       "B %s%s,C %s,alpha=%s,beta=%s)",
	       typeName<T>(), int(m), int(n), int(k), layoutName(lA),
	       conjA ? " conj" : "", layoutName(lB), conjB ? " conj" : "",
	       layoutName(lC), str(alpha).c_str(), str(beta).c_str());

    //  elements of C, and all other elements of its buffer unchanged
    std::vector<char> isC(C.buf.size(), 0);
    for (std::int64_t j = 0; j < n; ++j) {
	for (std::int64_t i = 0; i < m; ++i) {
	    isC[std::size_t(C.base + i * C.rs + j * C.cs)] = 1;
	    T t = fromCdouble<T>(ref[i + j * m]);
	    if (!ctx.check(closeEnough(C(i, j), t, bound[i + j * m],
	                               int(std::max<std::int64_t>(k, 1))),
	                   "%s: C(%d,%d) = %s, expected %s", what.c_str(),
	                   int(i), int(j), str(C(i, j)).c_str(),
	                   str(t).c_str())) {
		return false;
	    }
	}
    }
    for (std::size_t p = 0; p < C.buf.size(); ++p) {
	if (!isC[p] && !ctx.check(sameBits(C.buf[p], C0[p]),
	                          "%s: element %d of the buffer of C outside "
	                          "of C modified",
	                          what.c_str(), int(p))) {
	    return false;
	}
    }
    return true;
}

template <typename T>
T
scalar(double re, double im)
{
    return fromCdouble<T>(cdouble(re, im));
}

//
//  All layouts of A, B and C (C without zero strides) for one size
//
template <typename T>
void
testLayouts(Context &ctx, std::int64_t m, std::int64_t n, std::int64_t k)
{
    Random random(0x5172ed + m * 1000003 + n * 1009 + k);
    const T alphas[] = {scalar<T>(1, 0), scalar<T>(-1.5, 0.5)};
    const T betas[] = {scalar<T>(0, 0), scalar<T>(0.5, -0.25)};
    const bool cplx = IsComplex<T>::value;

    for (int lA = 0; lA < NumLayouts; ++lA) {
	for (int lB = 0; lB < NumLayouts; ++lB) {
	    for (int lC = 0; lC < ZeroRow; ++lC) {
		//  all combinations of A and B with column-major C; all
		//  layouts of C with A, B column major or negative
		if (lC != ColMajor && !(lA == ColMajor && lB == ColMajor) &&
		    !(lA == NegBoth && lB == Mixed)) {
		    continue;
		}
		for (int conj = 0; conj < (cplx ? 4 : 1); ++conj) {
		    T alpha = alphas[(lA + lB + conj) % 2];
		    T beta = betas[(lA + lC + conj) % 2];
		    if (!stridedCase<T>(ctx, random, m, n, k, Layout(lA),
		                        Layout(lB), Layout(lC), conj & 1,
		                        conj & 2, alpha, beta)) {
			return;
		    }
		}
	    }
	}
    }
}

template <typename T>
void
testSizes(Context &ctx)
{
    //  tiny, partial tiles and blocks of the test configurations, and
    //  sizes beyond the default blocks
    static const int size[][3] = {{1, 1, 1},     {2, 3, 1},    {7, 5, 3},
                                  {13, 17, 11},  {33, 31, 65}, {64, 64, 64},
                                  {97, 113, 301}};
    for (const auto &s : size) {
	testLayouts<T>(ctx, s[0], s[1], s[2]);
    }
}

template <typename T>
void
testLarge(Context &ctx)
{
    Random random(0x1a76e);
    const T alpha = scalar<T>(0.75, -0.5), beta = scalar<T>(-0.5, 0.25);
    const Layout layouts[][3] = {{ColMajor, ColMajor, ColMajor},
                                 {RowStride2, NegRow, RowMajor},
                                 {Mixed, NegBoth, RowStride2},
                                 {ZeroRow, RowMajor, NegRow}};
    for (const auto &l : layouts) {
	if (!stridedCase<T>(ctx, random, 413, 389, 1031, l[0], l[1], l[2],
	                    IsComplex<T>::value, false, alpha, beta)) {
	    return;
	}
    }
}

//
//  alpha = 0, k = 0, empty C, and zero strides of C where they are allowed
//  (dimension of length 1)
//
template <typename T>
void
testSpecial(Context &ctx)
{
    Random random(0x5e1a1);
    const T one = scalar<T>(1, 0), zero = scalar<T>(0, 0);
    const T beta = scalar<T>(-2, 0.5);
    struct
    {
	std::int64_t m, n, k;
	T alpha, beta;
    } c[] = {{9, 7, 5, zero, beta}, {9, 7, 5, zero, zero}, {9, 7, 0, one, beta},
             {9, 7, 0, one, zero},  {0, 7, 5, one, beta},  {9, 0, 5, one, beta},
             {0, 0, 0, one, zero}};
    for (const auto &x : c) {
	if (!stridedCase<T>(ctx, random, x.m, x.n, x.k, RowStride2, NegRow,
	                    Mixed, false, false, x.alpha, x.beta)) {
	    return;
	}
    }
    //  C a row vector with row stride 0, a column vector with column
    //  stride 0
    for (std::int64_t k : {1, 6, 300}) {
	if (!stridedCase<T>(ctx, random, 1, 37, k, Mixed, RowMajor, ZeroRow,
	                    false, IsComplex<T>::value, one, beta) ||
	    !stridedCase<T>(ctx, random, 37, 1, k, NegRow, ColMajor, ZeroCol,
	                    IsComplex<T>::value, false, one, zero)) {
	    return;
	}
    }
}

//
//  ulmBLAS::ger (C++ template) with a matrix A of which neither stride is 1:
//  A is packed into a buffer, updated and copied back.  The buffer must be
//  loaded with A first (it was not, and the result was the update added to
//  whatever the buffer held; found by getf2 with strided matrices).  Two
//  calls in a row, so that a reused buffer holds other data.
//
template <typename T>
bool
gerCaseStrided(Context &ctx, Random &random, std::int64_t m, std::int64_t n,
               std::int64_t rs, std::int64_t cs)
{
    const std::int64_t lo = std::min<std::int64_t>(0, (m - 1) * rs) +
                            std::min<std::int64_t>(0, (n - 1) * cs);
    const std::int64_t hi = std::max<std::int64_t>(0, (m - 1) * rs) +
                            std::max<std::int64_t>(0, (n - 1) * cs);
    std::vector<T> buf(hi - lo + 1), x(m), y(n);
    for (T &v : buf)
	v = random.value<T>();
    for (T &v : x)
	v = random.value<T>();
    for (T &v : y)
	v = random.value<T>();
    T *A = &buf[-lo];
    const T alpha[2] = {scalar<T>(0.75, -0.5), scalar<T>(-1.5, 0.25)};

    for (int call = 0; call < 2; ++call) {
	std::vector<T> ref(buf);
	T *R = &ref[-lo];
	for (std::int64_t j = 0; j < n; ++j) {
	    for (std::int64_t i = 0; i < m; ++i) {
		R[i * rs + j * cs] += alpha[call] * x[i] * y[j];
	    }
	}
	ulmBLAS::ger(m, n, alpha[call], x.data(), std::int64_t(1), y.data(),
	             std::int64_t(1), A, rs, cs);
	for (std::size_t k = 0; k < buf.size(); ++k) {
	    typedef decltype(std::abs(T(0))) Real;
	    const Real tol = 8 * std::numeric_limits<Real>::epsilon() *
	                     (std::abs(ref[k]) + 2);
	    if (!ctx.check(std::abs(buf[k] - ref[k]) <= tol,
	                   "ger m=%lld n=%lld rs=%lld cs=%lld call %d: "
	                   "element %zu wrong",
	                   (long long)m, (long long)n, (long long)rs,
	                   (long long)cs, call + 1, k)) {
		return false;
	    }
	}
    }
    return true;
}

template <typename T>
void
testGerStrided(Context &ctx)
{
    Random random(0x6e4b5);
    const std::int64_t size[][2] = {
        {1, 1}, {5, 3}, {37, 29}, {300, 260}, {700, 1100}};
    for (const auto &s : size) {
	const std::int64_t m = s[0], n = s[1];
	const std::int64_t st[][2] = {{2, 2 * m + 1},
	                              {3, 3 * m + 2},
	                              {2 * n + 1, 2},
	                              {-2, 2 * m + 1},
	                              {3, -(3 * m + 1)}};
	for (const auto &t : st) {
	    if (!gerCaseStrided<T>(ctx, random, m, n, t[0], t[1])) {
		return;
	    }
	}
    }
}

//
//  Integer GEMM (ulmblas_i{8,16,32,64}gemm_strided): exact result with
//  wraparound, compared with a plain loop in the unsigned type.  Values
//  over the whole range, so products and sums overflow.
//
void
gemmInt(std::int64_t m, std::int64_t n, std::int64_t k, std::int8_t alpha,
        const std::int8_t *A, std::int64_t rA, std::int64_t cA,
        const std::int8_t *B, std::int64_t rB, std::int64_t cB,
        std::int8_t beta, std::int8_t *C, std::int64_t rC, std::int64_t cC)
{
    ulmblas_i8gemm_strided(m, n, k, alpha, A, rA, cA, B, rB, cB, beta, C, rC,
                           cC);
}

void
gemmInt(std::int64_t m, std::int64_t n, std::int64_t k, std::int16_t alpha,
        const std::int16_t *A, std::int64_t rA, std::int64_t cA,
        const std::int16_t *B, std::int64_t rB, std::int64_t cB,
        std::int16_t beta, std::int16_t *C, std::int64_t rC, std::int64_t cC)
{
    ulmblas_i16gemm_strided(m, n, k, alpha, A, rA, cA, B, rB, cB, beta, C, rC,
                            cC);
}

void
gemmInt(std::int64_t m, std::int64_t n, std::int64_t k, std::int32_t alpha,
        const std::int32_t *A, std::int64_t rA, std::int64_t cA,
        const std::int32_t *B, std::int64_t rB, std::int64_t cB,
        std::int32_t beta, std::int32_t *C, std::int64_t rC, std::int64_t cC)
{
    ulmblas_i32gemm_strided(m, n, k, alpha, A, rA, cA, B, rB, cB, beta, C, rC,
                            cC);
}

void
gemmInt(std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t alpha,
        const std::int64_t *A, std::int64_t rA, std::int64_t cA,
        const std::int64_t *B, std::int64_t rB, std::int64_t cB,
        std::int64_t beta, std::int64_t *C, std::int64_t rC, std::int64_t cC)
{
    ulmblas_i64gemm_strided(m, n, k, alpha, A, rA, cA, B, rB, cB, beta, C, rC,
                            cC);
}

//  an r x c operand in a buffer of its own: (offset, row stride, col stride)
struct IntOperand
{
    std::int64_t off, rs, cs, size;
};

IntOperand
intOperand(int layout, std::int64_t r, std::int64_t c)
{
    r = r > 0 ? r : 1;
    c = c > 0 ? c : 1;
    switch (layout) {
    case 0:
	return {0, 1, r + 2, (r + 2) * c}; // col major
    case 1:
	return {0, c + 1, 1, (c + 1) * r}; // row major
    case 2:
	return {2 * (r - 1), -2, 2 * r + 1, (2 * r + 1) * c}; // negative
    default:
	return {0, 0, 1, c}; // zero row stride
    }
}

template <typename T>
bool
intCase(Context &ctx, std::uint64_t &seed, std::int64_t m, std::int64_t n,
        std::int64_t k, int lA, int lB, int lC, T alpha, T beta)
{
    typedef typename std::make_unsigned<T>::type U;
    auto next = [&seed]() -> U {
	seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
	return U(seed >> 17);
    };
    const IntOperand a = intOperand(lA, m, k), b = intOperand(lB, k, n);
    const IntOperand c = intOperand(lC == 3 ? 0 : lC, m, n);
    std::vector<T> A(std::size_t(a.size)), B(std::size_t(b.size)),
        C(std::size_t(c.size));
    for (T &x : A)
	x = T(next());
    for (T &x : B)
	x = T(next());
    for (T &x : C)
	x = T(next());
    const std::vector<T> C0 = C;

    gemmInt(m, n, k, alpha, A.data() + a.off, a.rs, a.cs, B.data() + b.off,
            b.rs, b.cs, beta, C.data() + c.off, c.rs, c.cs);

    std::vector<char> inside(C.size(), 0);
    bool ok = true;
    for (std::int64_t j = 0; j < n; ++j) {
	for (std::int64_t i = 0; i < m; ++i) {
	    U s = 0;
	    for (std::int64_t l = 0; l < k; ++l) {
		s = U(s + U(U(A[std::size_t(a.off + i * a.rs + l * a.cs)]) *
		            U(B[std::size_t(b.off + l * b.rs + j * b.cs)])));
	    }
	    const std::size_t q = std::size_t(c.off + i * c.rs + j * c.cs);
	    U ref = U(U(alpha) * s);
	    if (beta != 0) {
		ref = U(ref + U(U(beta) * U(C0[q])));
	    }
	    ok = ok && U(C[q]) == ref;
	    inside[q] = 1;
	}
    }
    for (std::size_t q = 0; q < C.size(); ++q) {
	ok = ok && (inside[q] || C[q] == C0[q]);
    }
    return ctx.check(ok,
                     "i%dgemm_strided m=%lld n=%lld k=%lld layouts %d %d "
                     "%d alpha=%lld beta=%lld",
                     int(8 * sizeof(T)), (long long)m, (long long)n,
                     (long long)k, lA, lB, lC, (long long)alpha,
                     (long long)beta);
}

template <typename T>
void
testInt(Context &ctx)
{
    std::uint64_t seed = 0x1d2c3b4a + sizeof(T);
    const std::int64_t size[][3] = {
        {1, 1, 1},       {7, 5, 3}, {13, 17, 29}, {70, 53, 131},
        {301, 263, 517}, {0, 5, 3}, {4, 0, 3},    {4, 5, 0}};
    const T ab[][2] = {{T(1), T(0)}, {T(-3), T(5)}, {T(0), T(7)}};
    int i = 0;
    for (const auto &s : size) {
	for (int lA = 0; lA < 4; ++lA) {
	    for (int lB = 0; lB < 4; ++lB) {
		const int lC = (lA + lB) % 3;
		const auto &x = ab[(i++) % 3];
		if (!intCase<T>(ctx, seed, s[0], s[1], s[2], lA, lB, lC, x[0],
		                x[1])) {
		    return;
		}
	    }
	}
    }
}

//
//  ulmblas_sbgemm_strided, ulmblas_shgemm_strided: A and B in bfloat16 or
//  half precision (uint16_t), C in float.  The reference in double from the
//  values of A and B; the products of two 16-bit values are exact in float,
//  so the error bound is that of sgemm.
//
void
gemm16(const ulmBLAS::bfloat16 *, std::int64_t m, std::int64_t n,
       std::int64_t k, float alpha, const std::uint16_t *A, std::int64_t rA,
       std::int64_t cA, const std::uint16_t *B, std::int64_t rB,
       std::int64_t cB, float beta, float *C, std::int64_t rC, std::int64_t cC)
{
    ulmblas_sbgemm_strided(m, n, k, alpha, A, rA, cA, B, rB, cB, beta, C, rC,
                           cC);
}

void
gemm16(const ulmBLAS::float16 *, std::int64_t m, std::int64_t n, std::int64_t k,
       float alpha, const std::uint16_t *A, std::int64_t rA, std::int64_t cA,
       const std::uint16_t *B, std::int64_t rB, std::int64_t cB, float beta,
       float *C, std::int64_t rC, std::int64_t cC)
{
    ulmblas_shgemm_strided(m, n, k, alpha, A, rA, cA, B, rB, cB, beta, C, rC,
                           cC);
}

template <typename T16>
float
toFloat(std::uint16_t bits)
{
    T16 x;
    x.bits = bits;
    return float(x);
}

template <typename T16>
bool
gemm16Case(Context &ctx, Random &random, std::int64_t m, std::int64_t n,
           std::int64_t k, Layout lA, Layout lB, Layout lC, float alpha,
           float beta)
{
    Operand<std::uint16_t> A(lA, m, k), B(lB, k, n);
    Operand<float> C(lC, m, n);

    for (auto &x : A.buf)
	x = T16(random.value<float>()).bits;
    for (auto &x : B.buf)
	x = T16(random.value<float>()).bits;
    randomize(random, C.buf);
    if (alpha == 0) {
	//  A and B must not be read
	for (auto &x : A.buf)
	    x = T16(nan<float>()).bits;
	for (auto &x : B.buf)
	    x = T16(nan<float>()).bits;
    }
    if (beta == 0) {
	//  C must not be read
	for (std::int64_t j = 0; j < n; ++j) {
	    for (std::int64_t i = 0; i < m; ++i) {
		C(i, j) = nan<float>();
	    }
	}
    }
    std::vector<float> C0 = C.buf;

    std::vector<double> ref(std::size_t(std::max<std::int64_t>(m * n, 1)));
    std::vector<double> bound(ref.size());
    for (std::int64_t j = 0; j < n; ++j) {
	for (std::int64_t i = 0; i < m; ++i) {
	    double s = 0, b = 0;
	    if (alpha != 0) {
		for (std::int64_t l = 0; l < k; ++l) {
		    double p = double(toFloat<T16>(A(i, l))) *
		               double(toFloat<T16>(B(l, j)));
		    s += p;
		    b += std::abs(p);
		}
	    }
	    double c = beta == 0 ? 0 : double(C(i, j));
	    ref[i + j * m] = alpha * s + beta * c;
	    bound[i + j * m] =
	        std::abs(alpha) * b + std::abs(beta) * std::abs(c);
	}
    }

    gemm16(static_cast<const T16 *>(nullptr), m, n, k, alpha, A.ptr(), A.rs,
           A.cs, B.ptr(), B.rs, B.cs, beta, C.ptr(), C.rs, C.cs);

    std::string what = format(
        "%s_strided(m=%d,n=%d,k=%d,A %s,B %s,C %s,alpha=%g,beta=%g)",
        std::is_same<T16, ulmBLAS::bfloat16>::value ? "sbgemm" : "shgemm",
        int(m), int(n), int(k), layoutName(lA), layoutName(lB), layoutName(lC),
        double(alpha), double(beta));

    std::vector<char> isC(C.buf.size(), 0);
    for (std::int64_t j = 0; j < n; ++j) {
	for (std::int64_t i = 0; i < m; ++i) {
	    isC[std::size_t(C.base + i * C.rs + j * C.cs)] = 1;
	    float t = float(ref[i + j * m]);
	    if (!ctx.check(closeEnough(C(i, j), t, bound[i + j * m],
	                               int(std::max<std::int64_t>(k, 1))),
	                   "%s: C(%d,%d) = %g, expected %g", what.c_str(),
	                   int(i), int(j), double(C(i, j)), double(t))) {
		return false;
	    }
	}
    }
    for (std::size_t p = 0; p < C.buf.size(); ++p) {
	if (!isC[p] && !ctx.check(sameBits(C.buf[p], C0[p]),
	                          "%s: element %d of the buffer of C outside "
	                          "of C modified",
	                          what.c_str(), int(p))) {
	    return false;
	}
    }
    return true;
}

template <typename T16>
void
testGemm16(Context &ctx)
{
    Random random(0xb16f16);
    static const int size[][3] = {{1, 1, 1},     {2, 3, 1},    {7, 5, 3},
                                  {13, 17, 11},  {33, 31, 65}, {64, 64, 64},
                                  {97, 113, 301}};
    const float alphas[] = {1.0f, -1.5f}, betas[] = {0.0f, 0.5f};

    for (const auto &s : size) {
	for (int lA = 0; lA < NumLayouts; ++lA) {
	    for (int lB = 0; lB < NumLayouts; ++lB) {
		for (int lC = 0; lC < ZeroRow; ++lC) {
		    if (lC != ColMajor && !(lA == ColMajor && lB == ColMajor) &&
		        !(lA == NegBoth && lB == Mixed)) {
			continue;
		    }
		    if (!gemm16Case<T16>(ctx, random, s[0], s[1], s[2],
		                         Layout(lA), Layout(lB), Layout(lC),
		                         alphas[(lA + lB) % 2],
		                         betas[(lA + lC) % 2])) {
			return;
		    }
		}
	    }
	}
    }
    //  large (several blocks), alpha = 0, k = 0, empty
    const Layout large[][3] = {{ColMajor, ColMajor, ColMajor},
                               {RowMajor, RowMajor, RowMajor},
                               {RowStride2, NegRow, RowMajor},
                               {ZeroRow, RowMajor, NegRow}};
    for (const auto &l : large) {
	if (!gemm16Case<T16>(ctx, random, 413, 389, 1031, l[0], l[1], l[2],
	                     0.75f, -0.5f)) {
	    return;
	}
    }
    struct
    {
	std::int64_t m, n, k;
	float alpha, beta;
    } c[] = {{9, 7, 5, 0, -2}, {9, 7, 5, 0, 0},  {9, 7, 0, 1, -2},
             {9, 7, 0, 1, 0},  {0, 7, 5, 1, -2}, {9, 0, 5, 1, -2}};
    for (const auto &x : c) {
	if (!gemm16Case<T16>(ctx, random, x.m, x.n, x.k, RowStride2, NegRow,
	                     Mixed, x.alpha, x.beta)) {
	    return;
	}
    }
}

} // namespace

ULMBLAS_TEST(T0800, "strided/sgemm all layouts")
{
    testSizes<float>(ctx);
}
ULMBLAS_TEST(T0801, "strided/dgemm all layouts")
{
    testSizes<double>(ctx);
}
ULMBLAS_TEST(T0802, "strided/cgemm all layouts")
{
    testSizes<cfloat>(ctx);
}
ULMBLAS_TEST(T0803, "strided/zgemm all layouts")
{
    testSizes<cdouble>(ctx);
}
ULMBLAS_TEST(T0804, "strided/?gemm large")
{
    testLarge<float>(ctx);
    testLarge<double>(ctx);
    testLarge<cfloat>(ctx);
    testLarge<cdouble>(ctx);
}
ULMBLAS_TEST(T0805, "strided/?gemm alpha=0, k=0, empty, zero strides of C")
{
    testSpecial<float>(ctx);
    testSpecial<double>(ctx);
    testSpecial<cfloat>(ctx);
    testSpecial<cdouble>(ctx);
}
ULMBLAS_TEST(T0806, "strided/ulmBLAS::ger, neither stride of A is 1")
{
    testGerStrided<float>(ctx);
    testGerStrided<double>(ctx);
    testGerStrided<cfloat>(ctx);
    testGerStrided<cdouble>(ctx);
}
ULMBLAS_TEST(T0807, "strided/i8, i16, i32, i64 gemm: exact with wraparound")
{
    testInt<std::int8_t>(ctx);
    testInt<std::int16_t>(ctx);
    testInt<std::int32_t>(ctx);
    testInt<std::int64_t>(ctx);
}
ULMBLAS_TEST(T0808, "strided/sbgemm, shgemm (bfloat16, half precision)")
{
    testGemm16<ulmBLAS::bfloat16>(ctx);
    testGemm16<ulmBLAS::float16>(ctx);
}
