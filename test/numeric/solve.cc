//
//  Tests of the solves and inverses of ulmBLAS (templates in ulmblas/lapack
//  and the C interface): ?getrs, ?gesv, ?potrs, ?posv, ?trtrs, ?trtri,
//  ?lauum, ?potri, ?getri for float, double, complex<float> and
//  complex<double>.
//
//  Matrices in buffers of their own with several storage layouts (column
//  and row major, row stride 2, neither stride 1); elements outside of the
//  operands must not change (bitwise).  Residuals in complex<double>.
//  Built with OpenMP, the larger cases (n = 300, nrhs = 400) run as tasks.
//
#include <algorithm>
#include <cmath>
#include <vector>
#include <ulmblas/ulmblas.h>
#include <ulmblas/lapack/gesv.h>
#include <ulmblas/lapack/getri.h>
#include <ulmblas/lapack/getrs.h>
#include <ulmblas/lapack/lauum.h>
#include <ulmblas/lapack/posv.h>
#include <ulmblas/lapack/potri.h>
#include <ulmblas/lapack/potrs.h>
#include <ulmblas/lapack/trtri.h>
#include <ulmblas/lapack/trtrs.h>
#include LAPACK_HEADER
#include <test/numeric/reference.h>
#include <test/numeric/testing.h>
#include <test/numeric/lapacktest.h>

using namespace test;
using namespace test::lapacktest;

namespace {

//  op(A)(i,j) for op = 0 (A), 1 (A^T), 2 (A^H)
inline cdouble
opElem(const Full &A, int op, int i, int j)
{
    return op == 0 ? A(i, j) : op == 1 ? A(j, i) : std::conj(A(j, i));
}

//  ||op(A) X - B|| / (||A|| ||X|| n eps), maximum norms
template <typename T>
double
residual(const Full &A, int op, Strided<T> &X, const Full &B)
{
    const int n = A.n, nrhs = B.n;
    double r = 0, a = 0, x = 0;
    for (int i = 0; i < n; ++i) {
	for (int j = 0; j < n; ++j) {
	    a = std::max(a, std::abs(A(i, j)));
	}
    }
    for (int j = 0; j < nrhs; ++j) {
	for (int i = 0; i < n; ++i) {
	    cdouble s = 0;
	    for (int l = 0; l < n; ++l) {
		s += opElem(A, op, i, l) * toCdouble(X(l, j));
	    }
	    r = std::max(r, std::abs(s - B(i, j)));
	    x = std::max(x, std::abs(toCdouble(X(i, j))));
	}
    }
    return r / (std::max(a, 1e-300) * std::max(x, 1e-300) * std::max(n, 1) *
                eps<T>());
}

//  the elements of the buffer outside of the m x n matrix are unchanged
template <typename T>
bool
gapsUnchanged(Strided<T> &A, const Strided<T> &before)
{
    std::vector<char> inside(A.buf.size(), 0);
    for (Index j = 0; j < A.n; ++j) {
	for (Index i = 0; i < A.m; ++i) {
	    inside[std::size_t(A.off + i * A.rs + j * A.cs)] = 1;
	}
    }
    for (std::size_t q = 0; q < A.buf.size(); ++q) {
	if (!inside[q] && !sameBits(A.buf[q], before.buf[q])) {
	    return false;
	}
    }
    return true;
}

//  Hermitian positive definite (n added to the diagonal)
template <typename T>
Full
hpd(Random &random, int n)
{
    Full F(n, n);
    for (int j = 0; j < n; ++j) {
	for (int i = j; i < n; ++i) {
	    cdouble x = toCdouble(random.value<T>());
	    if (i == j) {
		x = cdouble(x.real() + n, 0);
	    }
	    F(i, j) = x;
	    F(j, i) = std::conj(x);
	}
    }
    return F;
}

const Index sizes[][2] = {{1, 1}, {5, 3}, {40, 1}, {97, 17}, {300, 400}};

//  getrs (op N, T, C), gesv: residual, B unchanged outside, singular gesv
template <typename T>
void
testGetrs(Context &ctx)
{
    Random random(0x6e75);
    for (const auto &s : sizes) {
	const Index n = s[0], nrhs = s[1];
	for (const auto &la : layouts(n, n)) {
	    for (const auto &lb : layouts(n, nrhs)) {
		if (la != lb && n > 40) {
		    continue; // all combinations only for small n
		}
		Full A0{int(n), int(n)};
		Strided<T> A(n, n, la.first, la.second);
		fill(A0, A, random);
		std::vector<Index> piv(static_cast<std::size_t>(n));
		ulmBLAS::getrf(n, n, A.ptr(), A.rs, A.cs, piv.data(), Index(1));
		for (int op = 0; op < 3; ++op) {
		    Full B0{int(n), int(nrhs)};
		    Strided<T> B(n, nrhs, lb.first, lb.second);
		    fill(B0, B, random);
		    const Strided<T> before = B;
		    ulmBLAS::getrs(op > 0, op == 2, n, nrhs, A.ptr(), A.rs,
		                   A.cs, piv.data(), Index(1), B.ptr(), B.rs,
		                   B.cs);
		    const double r = residual(A0, op, B, B0);
		    if (!ctx.check(r < 30 && gapsUnchanged(B, before),
		                   "getrs op %d n=%lld nrhs=%lld A(%lld,%lld) "
		                   "B(%lld,%lld): residual %.3g",
		                   op, (long long)n, (long long)nrhs,
		                   (long long)A.rs, (long long)A.cs,
		                   (long long)B.rs, (long long)B.cs, r)) {
			return;
		    }
		}
		//  gesv on a fresh copy of A0
		Strided<T> C(n, n, la.first, la.second);
		for (Index j = 0; j < n; ++j) {
		    for (Index i = 0; i < n; ++i) {
			C(i, j) = fromCdouble<T>(A0(int(i), int(j)));
		    }
		}
		Full B0{int(n), int(nrhs)};
		Strided<T> B(n, nrhs, lb.first, lb.second);
		fill(B0, B, random);
		const Index info =
		    ulmBLAS::gesv(n, nrhs, C.ptr(), C.rs, C.cs, piv.data(),
		                  Index(1), B.ptr(), B.rs, B.cs);
		const double r = residual(A0, 0, B, B0);
		if (!ctx.check(info == 0 && r < 30,
		               "gesv n=%lld nrhs=%lld: info "
		               "%lld residual %.3g",
		               (long long)n, (long long)nrhs, (long long)info,
		               r)) {
		    return;
		}
	    }
	}
    }
    //  singular: zero column 2, B unchanged
    const Index n = 6;
    Full A0{int(n), int(n)};
    Strided<T> A(n, n, 1, n), B(n, 2, 1, n);
    Full B0{int(n), 2};
    fill(A0, A, random);
    fill(B0, B, random);
    for (Index i = 0; i < n; ++i) {
	A(i, 2) = T(0);
    }
    const Strided<T> before = B;
    std::vector<Index> piv(static_cast<std::size_t>(n));
    const Index info = ulmBLAS::gesv(n, Index(2), A.ptr(), A.rs, A.cs,
                                     piv.data(), Index(1), B.ptr(), B.rs, B.cs);
    bool same = true;
    for (std::size_t q = 0; q < B.buf.size(); ++q) {
	same = same && sameBits(B.buf[q], before.buf[q]);
    }
    ctx.check(info == 3 && same, "gesv singular: info %lld (expected 3), B %s",
              (long long)info, same ? "unchanged" : "changed");
}

//  potrs, posv (lower, upper)
template <typename T>
void
testPotrs(Context &ctx)
{
    Random random(0x9075);
    for (const auto &s : sizes) {
	const Index n = s[0], nrhs = s[1];
	for (int lower = 0; lower < 2; ++lower) {
	    for (const auto &la : layouts(n, n)) {
		const auto lb = layouts(n, nrhs)[std::size_t(n) % 4];
		const Full F = hpd<T>(random, int(n));
		Strided<T> A(n, n, la.first, la.second), C = A;
		for (Index j = 0; j < n; ++j) {
		    for (Index i = 0; i < n; ++i) {
			if (lower ? i >= j : i <= j) {
			    A(i, j) = fromCdouble<T>(F(int(i), int(j)));
			}
		    }
		}
		C = A;
		ulmBLAS::potrf(lower != 0, n, A.ptr(), A.rs, A.cs);
		Full B0{int(n), int(nrhs)};
		Strided<T> B(n, nrhs, lb.first, lb.second);
		fill(B0, B, random);
		Strided<T> B2 = B;
		const Strided<T> before = B;
		ulmBLAS::potrs(lower != 0, n, nrhs, A.ptr(), A.rs, A.cs,
		               B.ptr(), B.rs, B.cs);
		const double r = residual(F, 0, B, B0);
		const Index info =
		    ulmBLAS::posv(lower != 0, n, nrhs, C.ptr(), C.rs, C.cs,
		                  B2.ptr(), B2.rs, B2.cs);
		const double r2 = residual(F, 0, B2, B0);
		if (!ctx.check(r < 30 && r2 < 30 && info == 0 &&
		                   gapsUnchanged(B, before),
		               "potrs/posv %s n=%lld nrhs=%lld: residuals "
		               "%.3g %.3g, info %lld",
		               lower ? "lower" : "upper", (long long)n,
		               (long long)nrhs, r, r2, (long long)info)) {
		    return;
		}
	    }
	}
    }
}

//  a triangular matrix with a dominant diagonal (well conditioned)
template <typename T>
Full
triangular(Random &random, int n, bool lower, Strided<T> &A)
{
    Full A0(n, n);
    for (int j = 0; j < n; ++j) {
	for (int i = 0; i < n; ++i) {
	    if (lower ? i >= j : i <= j) {
		cdouble x = toCdouble(random.value<T>());
		if (i == j) {
		    x += (x.real() >= 0) ? 2.0 : -2.0;
		}
		A(i, j) = fromCdouble<T>(x);
		A0(i, j) = toCdouble(A(i, j));
	    }
	}
    }
    return A0;
}

//  trtrs (lower, upper; op N, T, C; unit, non-unit), INFO
template <typename T>
void
testTrtrs(Context &ctx)
{
    Random random(0x7275);
    for (const auto &s : sizes) {
	const Index n = s[0], nrhs = s[1];
	for (int lower = 0; lower < 2; ++lower) {
	    for (int unit = 0; unit < 2; ++unit) {
		for (int op = 0; op < 3; ++op) {
		    const auto la = layouts(n, n)[std::size_t(op + lower) % 4];
		    const auto lb =
		        layouts(n, nrhs)[std::size_t(op + unit) % 4];
		    Strided<T> A(n, n, la.first, la.second);
		    Full A0 = triangular(random, int(n), lower != 0, A);
		    if (unit) {
			for (int i = 0; i < int(n); ++i) {
			    A0(i, i) = 1;
			}
		    }
		    Full B0{int(n), int(nrhs)};
		    Strided<T> B(n, nrhs, lb.first, lb.second);
		    fill(B0, B, random);
		    const Strided<T> before = B;
		    const Index info = ulmBLAS::trtrs(
		        lower != 0, op > 0, op == 2, unit != 0, n, nrhs,
		        A.ptr(), A.rs, A.cs, B.ptr(), B.rs, B.cs);
		    const double r = residual(A0, op, B, B0);
		    if (!ctx.check(info == 0 && r < 30 &&
		                       gapsUnchanged(B, before),
		                   "trtrs %s %s op %d n=%lld nrhs=%lld: info "
		                   "%lld residual %.3g",
		                   lower ? "lower" : "upper",
		                   unit ? "unit" : "non-unit", op, (long long)n,
		                   (long long)nrhs, (long long)info, r)) {
			return;
		    }
		}
	    }
	}
    }
    //  zero diagonal element 3: INFO = 4, B unchanged
    Strided<T> A(5, 5, 1, 5), B(5, 1, 1, 5);
    triangular(random, 5, true, A);
    A(3, 3) = T(0);
    Full B0(5, 1);
    fill(B0, B, random);
    const Strided<T> before = B;
    const Index info =
        ulmBLAS::trtrs(true, false, false, false, Index(5), Index(1), A.ptr(),
	               A.rs, A.cs, B.ptr(), B.rs, B.cs);
    bool same = true;
    for (std::size_t q = 0; q < B.buf.size(); ++q) {
	same = same && sameBits(B.buf[q], before.buf[q]);
    }
    ctx.check(info == 4 && same, "trtrs singular: info %lld (expected 4)",
              (long long)info);
}

//  ||A X - I|| / (||A|| ||X|| n eps) for the full matrices A and X
inline double
inverseError(const Full &A, const Full &X, double e)
{
    const int n = A.n;
    double r = 0, a = 0, x = 0;
    for (int i = 0; i < n; ++i) {
	for (int j = 0; j < n; ++j) {
	    cdouble s = 0;
	    for (int l = 0; l < n; ++l) {
		s += A(i, l) * X(l, j);
	    }
	    r = std::max(r, std::abs(s - cdouble(i == j ? 1 : 0)));
	    a = std::max(a, std::abs(A(i, j)));
	    x = std::max(x, std::abs(X(i, j)));
	}
    }
    return r / (a * x * std::max(n, 1) * e);
}

const Index invSizes[] = {1, 2, 7, 33, 100, 257};

//  trtri (lower, upper, unit, non-unit), INFO
template <typename T>
void
testTrtri(Context &ctx)
{
    Random random(0x7472);
    for (Index n : invSizes) {
	for (int lower = 0; lower < 2; ++lower) {
	    for (int unit = 0; unit < 2; ++unit) {
		const auto la =
		    layouts(n, n)[std::size_t(lower + 2 * unit) % 4];
		Strided<T> A(n, n, la.first, la.second);
		Full A0 = triangular(random, int(n), lower != 0, A);
		if (unit) {
		    for (int i = 0; i < int(n); ++i) {
			A0(i, i) = 1;
		    }
		}
		const Strided<T> before = A;
		const Index info = ulmBLAS::trtri(lower != 0, unit != 0, n,
		                                  A.ptr(), A.rs, A.cs);
		Full X{int(n), int(n)};
		bool other = true;
		for (Index j = 0; j < n; ++j) {
		    for (Index i = 0; i < n; ++i) {
			const bool in = lower ? i >= j : i <= j;
			if (!in) {
			    other = other &&
			            sameBits(A(i, j),
			                     before.buf[std::size_t(
			                         A.off + i * A.rs + j * A.cs)]);
			} else {
			    X(int(i), int(j)) = (unit && i == j)
			                            ? cdouble(1)
			                            : toCdouble(A(i, j));
			}
		    }
		}
		const double r = inverseError(A0, X, eps<T>());
		if (!ctx.check(info == 0 && r < 30 && other,
		               "trtri %s %s n=%lld: info %lld error %.3g%s",
		               lower ? "lower" : "upper",
		               unit ? "unit" : "non-unit", (long long)n,
		               (long long)info, r,
		               other ? "" : ", other triangle changed")) {
		    return;
		}
	    }
	}
    }
    Strided<T> A(4, 4, 1, 4);
    triangular(random, 4, false, A);
    A(2, 2) = T(0);
    const Strided<T> before = A;
    const Index info =
        ulmBLAS::trtri(false, false, Index(4), A.ptr(), A.rs, A.cs);
    bool same = true;
    for (std::size_t q = 0; q < A.buf.size(); ++q) {
	same = same && sameBits(A.buf[q], before.buf[q]);
    }
    ctx.check(info == 3 && same, "trtri singular: info %lld (expected 3), A %s",
              (long long)info, same ? "unchanged" : "changed");
}

//  lauum (lower: L^H L, upper: U U^H, real diagonal)
template <typename T>
void
testLauum(Context &ctx)
{
    Random random(0x6c61);
    for (Index n : invSizes) {
	for (int lower = 0; lower < 2; ++lower) {
	    const auto la = layouts(n, n)[std::size_t(n + lower) % 4];
	    Strided<T> A(n, n, la.first, la.second);
	    Full M = triangular(random, int(n), lower != 0, A);
	    for (Index i = 0; i < n; ++i) {
		A(i, i) = fromCdouble<T>(cdouble(M(int(i), int(i)).real(), 0));
		M(int(i), int(i)) = toCdouble(A(i, i));
	    }
	    ulmBLAS::lauum(lower != 0, n, A.ptr(), A.rs, A.cs);
	    double r = 0, s = 0;
	    for (Index i = 0; i < n; ++i) {
		for (Index j = 0; j < n; ++j) {
		    if (lower ? i < j : i > j) {
			continue;
		    }
		    cdouble e = 0;
		    for (Index l = 0; l < n; ++l) {
			e += lower ? std::conj(M(int(l), int(i))) *
			                 M(int(l), int(j))
			           : M(int(i), int(l)) *
			                 std::conj(M(int(j), int(l)));
		    }
		    r = std::max(r, std::abs(e - toCdouble(A(i, j))));
		    s = std::max(s, std::abs(e));
		}
	    }
	    const double ratio = r / (s * double(n) * eps<T>());
	    if (!ctx.check(ratio < 30, "lauum %s n=%lld: error %.3g",
	                   lower ? "lower" : "upper", (long long)n, ratio)) {
		return;
	    }
	}
    }
}

//  potri (lower, upper), getri: A inv(A) = I
template <typename T>
void
testInverse(Context &ctx)
{
    Random random(0x696e);
    for (Index n : invSizes) {
	for (int lower = 0; lower < 2; ++lower) {
	    const auto la = layouts(n, n)[std::size_t(n + lower) % 4];
	    const Full F = hpd<T>(random, int(n));
	    Strided<T> A(n, n, la.first, la.second);
	    for (Index j = 0; j < n; ++j) {
		for (Index i = 0; i < n; ++i) {
		    if (lower ? i >= j : i <= j) {
			A(i, j) = fromCdouble<T>(F(int(i), int(j)));
		    }
		}
	    }
	    ulmBLAS::potrf(lower != 0, n, A.ptr(), A.rs, A.cs);
	    const Index info =
	        ulmBLAS::potri(lower != 0, n, A.ptr(), A.rs, A.cs);
	    Full X{int(n), int(n)};
	    for (Index j = 0; j < n; ++j) {
		for (Index i = 0; i < n; ++i) {
		    const bool in = lower ? i >= j : i <= j;
		    X(int(i), int(j)) =
		        in ? toCdouble(A(i, j)) : std::conj(toCdouble(A(j, i)));
		}
	    }
	    const double r = inverseError(F, X, eps<T>());
	    if (!ctx.check(info == 0 && r < 30,
	                   "potri %s n=%lld: info %lld error "
	                   "%.3g",
	                   lower ? "lower" : "upper", (long long)n,
	                   (long long)info, r)) {
		return;
	    }
	}
	const auto la = layouts(n, n)[std::size_t(n) % 4];
	Full A0{int(n), int(n)};
	Strided<T> A(n, n, la.first, la.second);
	fill(A0, A, random);
	std::vector<Index> piv(static_cast<std::size_t>(n));
	ulmBLAS::getrf(n, n, A.ptr(), A.rs, A.cs, piv.data(), Index(1));
	const Index info =
	    ulmBLAS::getri(n, A.ptr(), A.rs, A.cs, piv.data(), Index(1));
	Full X{int(n), int(n)};
	for (Index j = 0; j < n; ++j) {
	    for (Index i = 0; i < n; ++i) {
		X(int(i), int(j)) = toCdouble(A(i, j));
	    }
	}
	const double r = inverseError(A0, X, eps<T>());
	if (!ctx.check(info == 0 && r < 1000,
	               "getri n=%lld: info %lld error %.3g", (long long)n,
	               (long long)info, r)) {
	    return;
	}
    }
}

//
//  C interface: getrs/gesv/getri in row and column major (row major with
//  the factorization of ?getrf for row major), potrs/posv/potri/trtrs/
//  trtri/lauum for one case each, invalid arguments.
//
typedef ULMBLAS_INDEX_TYPE CIndex;

#define C_FUNCTIONS(P, T)                                                      \
    CIndex cGetrf(CBLAS_ORDER o, CIndex m, CIndex n, T *A, CIndex ld,          \
                  CIndex *p)                                                   \
    {                                                                          \
	return ULMLAPACK(P##getrf)(o, m, n, A, ld, p);                         \
    }                                                                          \
    CIndex cGesv(CBLAS_ORDER o, CIndex n, CIndex r, T *A, CIndex ld,           \
                 CIndex *p, T *B, CIndex ldb)                                  \
    {                                                                          \
	return ULMLAPACK(P##gesv)(o, n, r, A, ld, p, B, ldb);                  \
    }                                                                          \
    CIndex cGetrs(CBLAS_ORDER o, CBLAS_TRANSPOSE t, CIndex n, CIndex r,        \
                  const T *A, CIndex ld, const CIndex *p, T *B, CIndex ldb)    \
    {                                                                          \
	return ULMLAPACK(P##getrs)(o, t, n, r, A, ld, p, B, ldb);              \
    }                                                                          \
    CIndex cGetri(CBLAS_ORDER o, CIndex n, T *A, CIndex ld, const CIndex *p)   \
    {                                                                          \
	return ULMLAPACK(P##getri)(o, n, A, ld, p);                            \
    }                                                                          \
    CIndex cPosv(CBLAS_ORDER o, CBLAS_UPLO u, CIndex n, CIndex r, T *A,        \
                 CIndex ld, T *B, CIndex ldb)                                  \
    {                                                                          \
	return ULMLAPACK(P##posv)(o, u, n, r, A, ld, B, ldb);                  \
    }                                                                          \
    CIndex cPotri(CBLAS_ORDER o, CBLAS_UPLO u, CIndex n, T *A, CIndex ld)      \
    {                                                                          \
	return ULMLAPACK(P##potri)(o, u, n, A, ld);                            \
    }                                                                          \
    CIndex cTrtrs(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t,              \
                  CBLAS_DIAG d, CIndex n, CIndex r, const T *A, CIndex ld,     \
                  T *B, CIndex ldb)                                            \
    {                                                                          \
	return ULMLAPACK(P##trtrs)(o, u, t, d, n, r, A, ld, B, ldb);           \
    }

C_FUNCTIONS(s, float)
C_FUNCTIONS(d, double)
C_FUNCTIONS(c, cfloat)
C_FUNCTIONS(z, cdouble)

template <typename T>
void
testSolveC(Context &ctx)
{
    Random random(0xc501);
    const CIndex n = 37, nrhs = 5, ld = n + 2;
    for (int row = 0; row < 2; ++row) {
	const CBLAS_ORDER o = row ? CblasRowMajor : CblasColMajor;
	//  A(i,j) = a[i*ars + j*acs], B(i,j) = b[i*brs + j*bcs]
	const Index ars = row ? ld : 1, acs = row ? 1 : ld;
	const CIndex ldb = row ? nrhs + 1 : n + 1;
	const Index brs = row ? ldb : 1, bcs = row ? 1 : ldb;
	std::vector<T> a(std::size_t(ld * n)),
	    b(std::size_t(ldb * (row ? n : nrhs)));
	Full A0{int(n), int(n)}, B0{int(n), int(nrhs)};
	for (Index j = 0; j < n; ++j) {
	    for (Index i = 0; i < n; ++i) {
		a[std::size_t(i * ars + j * acs)] = random.value<T>();
		A0(int(i), int(j)) =
		    toCdouble(a[std::size_t(i * ars + j * acs)]);
	    }
	}
	for (Index j = 0; j < nrhs; ++j) {
	    for (Index i = 0; i < n; ++i) {
		b[std::size_t(i * brs + j * bcs)] = random.value<T>();
		B0(int(i), int(j)) =
		    toCdouble(b[std::size_t(i * brs + j * bcs)]);
	    }
	}
	std::vector<T> a2 = a, b2 = b;
	std::vector<CIndex> p(static_cast<std::size_t>(n));
	//  gesv
	const CIndex info =
	    cGesv(o, n, nrhs, a.data(), ld, p.data(), b.data(), ldb);
	Strided<T> X(n, nrhs, brs, bcs);
	auto toX = [&](const std::vector<T> &v) {
	    for (Index j = 0; j < nrhs; ++j) {
		for (Index i = 0; i < n; ++i) {
		    X(i, j) = v[std::size_t(i * brs + j * bcs)];
		}
	    }
	};
	toX(b);
	double r = residual(A0, 0, X, B0);
	if (!ctx.check(info == 0 && r < 30, "C ?gesv %s: info %d residual %.3g",
	               row ? "row" : "col", int(info), r)) {
	    return;
	}
	//  getrf + getrs with A^T and A^H
	cGetrf(o, n, n, a2.data(), ld, p.data());
	for (int op = 1; op < 3; ++op) {
	    std::vector<T> b3 = b2;
	    cGetrs(o, op == 1 ? CblasTrans : CblasConjTrans, n, nrhs, a2.data(),
	           ld, p.data(), b3.data(), ldb);
	    toX(b3);
	    r = residual(A0, op, X, B0);
	    if (!ctx.check(r < 30, "C ?getrs %s op %d: residual %.3g",
	                   row ? "row" : "col", op, r)) {
		return;
	    }
	}
	//  getri
	cGetri(o, n, a2.data(), ld, p.data());
	Full Xi{int(n), int(n)};
	for (Index j = 0; j < n; ++j) {
	    for (Index i = 0; i < n; ++i) {
		Xi(int(i), int(j)) =
		    toCdouble(a2[std::size_t(i * ars + j * acs)]);
	    }
	}
	r = inverseError(A0, Xi, eps<T>());
	if (!ctx.check(r < 1000, "C ?getri %s: error %.3g", row ? "row" : "col",
	               r)) {
	    return;
	}
	//  posv and potri (lower and upper)
	const Full F = hpd<T>(random, int(n));
	for (int lower = 0; lower < 2; ++lower) {
	    std::vector<T> h(std::size_t(ld * n), T(0)), hb = b2;
	    for (Index j = 0; j < n; ++j) {
		for (Index i = 0; i < n; ++i) {
		    h[std::size_t(i * ars + j * acs)] =
		        fromCdouble<T>(F(int(i), int(j)));
		}
	    }
	    const CBLAS_UPLO u = lower ? CblasLower : CblasUpper;
	    const CIndex i1 =
	        cPosv(o, u, n, nrhs, h.data(), ld, hb.data(), ldb);
	    toX(hb);
	    r = residual(F, 0, X, B0);
	    const CIndex i2 = cPotri(o, u, n, h.data(), ld);
	    Full Xp{int(n), int(n)};
	    for (Index j = 0; j < n; ++j) {
		for (Index i = 0; i < n; ++i) {
		    const bool in = lower ? i >= j : i <= j;
		    Xp(int(i), int(j)) =
		        in ? toCdouble(h[std::size_t(i * ars + j * acs)])
			   : std::conj(
		                 toCdouble(h[std::size_t(j * ars + i * acs)]));
		}
	    }
	    const double r2 = inverseError(F, Xp, eps<T>());
	    if (!ctx.check(i1 == 0 && i2 == 0 && r < 30 && r2 < 30,
	                   "C ?posv/?potri %s %s: residual %.3g error %.3g",
	                   row ? "row" : "col", lower ? "lower" : "upper", r,
	                   r2)) {
		return;
	    }
	}
    }
    //  invalid arguments
    T a[4] = {T(1), T(0), T(0), T(1)}, b[2] = {T(1), T(1)};
    CIndex p[2] = {0, 1};
    const CBLAS_ORDER bad = CBLAS_ORDER(42);
    ctx.check(cGesv(bad, 2, 1, a, 2, p, b, 2) == -1, "C ?gesv: order");
    ctx.check(cGesv(CblasColMajor, 2, 1, a, 2, p, b, 1) == -8, "C ?gesv: ldB");
    ctx.check(cGetrs(CblasColMajor, CBLAS_TRANSPOSE(42), 2, 1, a, 2, p, b, 2) ==
                  -2,
              "C ?getrs: trans");
    ctx.check(cGetrs(CblasColMajor, CblasNoTrans, 2, -1, a, 2, p, b, 2) == -4,
              "C ?getrs: nrhs < 0");
    ctx.check(cGetri(CblasColMajor, 2, a, 1, p) == -4, "C ?getri: ldA");
    ctx.check(cPosv(CblasColMajor, CBLAS_UPLO(42), 2, 1, a, 2, b, 2) == -2,
              "C ?posv: upLo");
    ctx.check(cTrtrs(CblasColMajor, CblasLower, CblasNoTrans, CBLAS_DIAG(42), 2,
                     1, a, 2, b, 2) == -4,
              "C ?trtrs: diag");
}

} // namespace

ULMBLAS_TEST(T0418, "lapack/ulmBLAS::getrs, gesv")
{
    testGetrs<float>(ctx);
    testGetrs<double>(ctx);
    testGetrs<cfloat>(ctx);
    testGetrs<cdouble>(ctx);
}

ULMBLAS_TEST(T0419, "lapack/ulmBLAS::potrs, posv")
{
    testPotrs<float>(ctx);
    testPotrs<double>(ctx);
    testPotrs<cfloat>(ctx);
    testPotrs<cdouble>(ctx);
}

ULMBLAS_TEST(T0420, "lapack/ulmBLAS::trtrs")
{
    testTrtrs<float>(ctx);
    testTrtrs<double>(ctx);
    testTrtrs<cfloat>(ctx);
    testTrtrs<cdouble>(ctx);
}

ULMBLAS_TEST(T0421, "lapack/ulmBLAS::trtri")
{
    testTrtri<float>(ctx);
    testTrtri<double>(ctx);
    testTrtri<cfloat>(ctx);
    testTrtri<cdouble>(ctx);
}

ULMBLAS_TEST(T0422, "lapack/ulmBLAS::lauum")
{
    testLauum<float>(ctx);
    testLauum<double>(ctx);
    testLauum<cfloat>(ctx);
    testLauum<cdouble>(ctx);
}

ULMBLAS_TEST(T0423, "lapack/ulmBLAS::potri, getri")
{
    testInverse<float>(ctx);
    testInverse<double>(ctx);
    testInverse<cfloat>(ctx);
    testInverse<cdouble>(ctx);
}

ULMBLAS_TEST(T0424,
             "lapack/C interface: solves and inverses, row and column major")
{
    testSolveC<float>(ctx);
    testSolveC<double>(ctx);
    testSolveC<cfloat>(ctx);
    testSolveC<cdouble>(ctx);
}
