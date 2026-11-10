//
//  Floating-point exception flags (fenv.h) after calls of the level 3
//  routines (see issue #0031): with operands whose products and sums are
//  exact (small integers, a diagonal of 2 for trsm) no flag may be raised,
//  and flags raised before the call must still be raised afterwards.
//
//  Entering and leaving streaming mode (SME kernels, Apple M4, M5) sets all
//  cumulative exception flags; without restoring FPSR every call above the
//  size of the direct path left invalid, division by zero, overflow,
//  underflow and inexact raised (NumPy: "divide by zero encountered in
//  matmul").  On other platforms the test checks the same conditions.
//
#include <cfenv>
#include <complex>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <test/numeric/blas.h>
#include <test/numeric/testing.h>

namespace {

using cd = std::complex<double>;
using cf = std::complex<float>;

//  A, B, C: entries 0 or 1 (C: 1), diagonal of A 2; D: diagonal matrix 2
template <typename T> struct Exact
{
    explicit Exact(int n)
        : n(n), A(n * n), B(n * n), C(n * n, T(1)), D(n * n, T(0))
    {
	for (int j = 0; j < n; ++j) {
	    for (int i = 0; i < n; ++i) {
		A[i + j * n] = T((i + 2 * j) % 3 == 0 ? 1 : 0);
		B[i + j * n] = T((2 * i + j) % 5 < 2 ? 1 : 0);
	    }
	    A[j + j * n] = T(2);
	    D[j + j * n] = T(2);
	}
    }

    void
    reset()
    {
	C.assign(C.size(), T(1));
    }

    int n;
    std::vector<T> A, B, C, D;
};

std::vector<std::pair<std::string, std::function<void()>>>
cases(int n)
{
    static const cd zone = 1, zzero = 0;
    static const cf cone = 1, czero = 0;
    auto d = std::make_shared<Exact<double>>(n);
    auto s = std::make_shared<Exact<float>>(n);
    auto z = std::make_shared<Exact<cd>>(n);
    auto c = std::make_shared<Exact<cf>>(n);
    std::vector<std::pair<std::string, std::function<void()>>> v;
    const std::string N = " n=" + std::to_string(n);

    using blas::mp;
    for (int beta = 0; beta < 2; ++beta) {
	const std::string b = beta ? ", beta=1" : ", beta=0";
	v.push_back({"dgemm" + N + b, [=] {
	                 d->reset();
	                 cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
			             n, n, n, 1, d->A.data(), n, d->B.data(), n,
			             beta, d->C.data(), n);
	             }});
	v.push_back({"dgemm alpha=2, A^T" + N + b, [=] {
	                 d->reset();
	                 cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n,
			             n, n, 2, d->A.data(), n, d->B.data(), n,
			             beta, d->C.data(), n);
	             }});
	v.push_back({"sgemm" + N + b, [=] {
	                 s->reset();
	                 cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, n,
			             n, n, 1, s->A.data(), n, s->B.data(), n,
			             beta, s->C.data(), n);
	             }});
	v.push_back({"zgemm" + N + b, [=] {
	                 z->reset();
	                 cblas_zgemm(
	                     CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n,
	                     mp(&zone), mp(z->A.data()), n, mp(z->B.data()), n,
	                     beta ? mp(&zone) : mp(&zzero), mp(z->C.data()), n);
	             }});
	v.push_back({"cgemm" + N + b, [=] {
	                 c->reset();
	                 cblas_cgemm(
	                     CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
	                     mp(&cone), mp(c->A.data()), n, mp(c->B.data()), n,
	                     beta ? mp(&cone) : mp(&czero), mp(c->C.data()), n);
	             }});
    }
    v.push_back({"dsymm" + N, [=] {
	             d->reset();
	             cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, n, n, 1,
		                 d->A.data(), n, d->B.data(), n, 1, d->C.data(),
		                 n);
                 }});
    v.push_back({"dsyrk" + N, [=] {
	             d->reset();
	             cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, n, n,
		                 1, d->A.data(), n, 1, d->C.data(), n);
                 }});
    v.push_back({"zherk" + N, [=] {
	             z->reset();
	             cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans, n, n,
		                 1, mp(z->A.data()), n, 1, mp(z->C.data()), n);
                 }});
    v.push_back({"dtrmm" + N, [=] {
	             d->reset();
	             cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower,
		                 CblasNoTrans, CblasNonUnit, n, n, 1,
		                 d->A.data(), n, d->C.data(), n);
                 }});
    v.push_back({"dtrsm" + N, [=] {
	             d->reset();
	             cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper,
		                 CblasNoTrans, CblasNonUnit, n, n, 1,
		                 d->D.data(), n, d->C.data(), n);
                 }});
    v.push_back({"ztrsm" + N, [=] {
	             z->reset();
	             cblas_ztrsm(CblasColMajor, CblasRight, CblasLower,
		                 CblasNoTrans, CblasNonUnit, n, n, mp(&zone),
		                 mp(z->D.data()), n, mp(z->C.data()), n);
                 }});
    return v;
}

std::string
names(int f)
{
    std::string s;
    if (f & FE_INVALID)
	s += " invalid";
    if (f & FE_DIVBYZERO)
	s += " divbyzero";
    if (f & FE_OVERFLOW)
	s += " overflow";
    if (f & FE_UNDERFLOW)
	s += " underflow";
    if (f & FE_INEXACT)
	s += " inexact";
    return s.empty() ? " none" : s;
}

} // namespace

ULMBLAS_TEST(T0359,
             "interface/level 3 routines with exact operands raise no "
             "floating-point exception flags and keep flags raised before "
             "(#0031)")
{
    const int kept = FE_UNDERFLOW | FE_INEXACT;
    bool ok = true;

    for (int n : {8, 40, 300}) {
	for (auto &c : cases(n)) {
	    std::feclearexcept(FE_ALL_EXCEPT);
	    c.second();
	    const int after = std::fetestexcept(FE_ALL_EXCEPT);
	    ok = ctx.check(after == 0, "%s: flags raised:%s", c.first.c_str(),
	                   names(after).c_str()) &&
	         ok;

	    std::feclearexcept(FE_ALL_EXCEPT);
	    std::feraiseexcept(kept);
	    c.second();
	    const int kept_after = std::fetestexcept(FE_ALL_EXCEPT);
	    ok = ctx.check(kept_after == kept,
	                   "%s: flags raised before "
	                   "(underflow inexact), after:%s",
	                   c.first.c_str(), names(kept_after).c_str()) &&
	         ok;
	}
    }
    (void)ok;
}
