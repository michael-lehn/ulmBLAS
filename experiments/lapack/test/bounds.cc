// Every kernel call shape of the task graphs: the output operand is a
// view inside its own array (margin around it, filled with random data);
// after the call nothing outside the view (or outside the lower triangle,
// for sylrk and the Cholesky kernel) may have changed, bitwise.
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "kernels.h"
using namespace ulmlapack;
std::mt19937_64 rng(3);
double
ur()
{
    return std::uniform_real_distribution<double>(-1, 1)(rng);
}
long bad = 0, tests = 0;

struct Out
{ // rows x cols view, strides (rs,cs), margin M
    int rows, cols, M, ld;
    bool rowmajor;
    std::vector<double> a, snap;
    Out(int r, int c, bool rm, int M_ = 9)
        : rows(r), cols(c), M(M_), rowmajor(rm)
    {
	int R = rm ? c : r, C = rm ? r : c;
	ld = R + 2 * M;
	a.resize(size_t(ld) * (C + 2 * M));
	for (double &x : a)
	    x = ur();
    }
    Index
    rs() const
    {
	return rowmajor ? ld : 1;
    }
    Index
    cs() const
    {
	return rowmajor ? 1 : ld;
    }
    double *
    p()
    {
	return &a[M + size_t(M) * ld];
    }
    void
    take()
    {
	snap = a;
    }
    // allowed(i,j): may element (i,j) of the view change
    template <typename F>
    void
    verify(const char *what, F allowed)
    {
	++tests;
	std::vector<char> ok(a.size(), 0);
	for (int j = 0; j < cols; ++j)
	    for (int i = 0; i < rows; ++i)
		if (allowed(i, j))
		    ok[(p() - &a[0]) + i * rs() + j * cs()] = 1;
	for (size_t x = 0; x < a.size(); ++x)
	    if (!ok[x] && std::memcmp(&a[x], &snap[x], 8)) {
		if (++bad < 10)
		    printf("%s %dx%d %s: element outside changed\n", what, rows,
		           cols, rowmajor ? "T" : "N");
		return;
	    }
    }
};
std::vector<double>
mat(int r, int c, bool spd = false)
{
    std::vector<double> v(size_t(r) * c);
    for (double &x : v)
	x = ur();
    if (spd)
	for (int i = 0; i < std::min(r, c); ++i)
	    v[i + size_t(i) * r] = r + 2;
    return v;
}
auto all = [](int, int) { return true; };
auto lower = [](int i, int j) { return i >= j; };

int
main()
{
    int sizes[] = {1,  2,  3,  5,  7,  8,  9,  12, 15, 16,  17,
                   24, 31, 32, 33, 48, 63, 64, 65, 96, 100, 128};
    for (int t = 0; t < 2; ++t) {
	const bool T = t; // transposed views: the upper variant of potrf
	for (int a : sizes)
	    for (int b : sizes) {
		const int c = sizes[(a * 7 + b) % 22];
		// potrf TRSM: trlsm(a, b, L a x a, B^T) ; output Aik is b x a
		{
		    std::vector<double> Lk = mat(a, a, true);
		    Out B(b, a, T);
		    B.take();
		    // L as a view with the same orientation: element (i,j) at
		    // Lk[i*lrs + j*lcs]
		    Index lrs = T ? a : 1, lcs = T ? 1 : a;
		    B.take();
		    ulmBLAS::trlsm(Index(a), Index(b), 1.0, false, false,
		                   Lk.data(), lrs, lcs, B.p(), B.cs(), B.rs());
		    B.verify("trlsm(potrf)", all);
		}
		// potrf SYRK: sylrk(b, a, A b x a, C b x b lower)
		{
		    std::vector<double> A = mat(b, a);
		    Index ars = T ? a : 1, acs = T ? 1 : b;
		    Out C(b, b, T);
		    C.take();
		    ulmBLAS::sylrk(Index(b), Index(a), -1.0, A.data(), ars, acs,
		                   1.0, C.p(), C.rs(), C.cs());
		    C.verify("sylrk", lower);
		}
		// potrf GEMM: gemm(b, c, a, A b x a, Ajk^T (Ajk c x a), C b x
		// c)
		{
		    std::vector<double> A = mat(b, a), Bm = mat(c, a);
		    Index ars = T ? a : 1, acs = T ? 1 : b, brs = T ? a : 1,
		          bcs = T ? 1 : c;
		    Out C(b, c, T);
		    C.take();
		    ulmBLAS::gemm(Index(b), Index(c), Index(a), -1.0, false,
		                  A.data(), ars, acs, false, Bm.data(), bcs,
		                  brs, 1.0, C.p(), C.rs(), C.cs());
		    C.verify("gemm(potrf)", all);
		}
		if (!T) {
		    // getrf SWAP_TRSM: trlsm unit (a, b), B a x b
		    {
			std::vector<double> Lk = mat(a, a, true);
			Out B(a, b, false);
			B.take();
			ulmBLAS::trlsm(Index(a), Index(b), 1.0, false, true,
			               Lk.data(), Index(1), Index(a), B.p(),
			               B.rs(), B.cs());
			B.verify("trlsm(getrf)", all);
		    }
		    // getrf GEMM NN: gemm(b, c, a)
		    {
			std::vector<double> A = mat(b, a), Bm = mat(a, c);
			Out C(b, c, false);
			C.take();
			ulmBLAS::gemm(Index(b), Index(c), Index(a), -1.0, false,
			              A.data(), Index(1), Index(b), false,
			              Bm.data(), Index(1), Index(a), 1.0, C.p(),
			              C.rs(), C.cs());
			C.verify("gemm(getrf)", all);
		    }
		    // panel: getrf_rec (a+b) x a, and laswp
		    {
			const int m = a + b;
			Out P(m, a, false);
			P.take();
			std::vector<Index> piv(std::min(m, a));
			getrf_rec(m, a, P.p(), P.rs(), P.cs(), piv.data());
			P.verify("getrf_rec", all);
			Out S(m, c, false);
			S.take();
			laswp(c, S.p(), S.rs(), S.cs(), 0, Index(piv.size()),
			      piv.data());
			S.verify("laswp", all);
		    }
		}
	    }
	for (int a :
	     sizes) { // diagonal tile: potrf_lower_rec, lower triangle only
	    Out P(a, a, T);
	    for (int i = 0; i < a; ++i)
		P.p()[i * P.rs() + i * P.cs()] = a + 2;
	    P.take();
	    potrf_lower_rec(Index(a), P.p(), P.rs(), P.cs());
	    P.verify("potrf_lower_rec", lower);
	}
    }
    printf("bounds: %ld tests, %ld violations\n", tests, bad);
    return bad != 0;
}
