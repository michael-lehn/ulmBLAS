// Two ways to multiply complex numbers: operator* of std::complex, which
// follows Annex G of C (infinities are recovered from NaN results, for GCC
// and clang a call of __muldc3), and the formula written out, as Fortran
// and the BLAS compute it.
#include <chrono>
#include <complex>
#include <cstdio>
#include <limits>

typedef std::complex<double> dcomplex;

dcomplex
times(const dcomplex &a, const dcomplex &b)
{
    return dcomplex(a.real() * b.real() - a.imag() * b.imag(),
                    a.real() * b.imag() + a.imag() * b.real());
}

void
scalStd(long n, const dcomplex &alpha, dcomplex *x)
{
    for (long i = 0; i < n; ++i) {
	x[i] = alpha * x[i];
    }
}

void
scalFormula(long n, const dcomplex &alpha, dcomplex *x)
{
    for (long i = 0; i < n; ++i) {
	x[i] = times(alpha, x[i]);
    }
}

template <typename F>
double
seconds(F f)
{
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

int
main()
{
    const double inf = std::numeric_limits<double>::infinity();
    const dcomplex a(inf, inf), b(1, 0);
    const dcomplex p = a * b, q = times(a, b);
    std::printf("(inf,inf) * (1,0): operator* (%g,%g), formula (%g,%g)\n",
                p.real(), p.imag(), q.real(), q.imag());

    const long n = 1000; // a constant: x can be an array
    const int reps = 1000000;
    dcomplex x[n];
    for (long i = 0; i < n; ++i) {
	x[i] = dcomplex(1, 0);
    }
    const dcomplex alpha(0.6, 0.8); // |alpha| = 1: no overflow
    const double tStd = seconds([&] {
	for (int r = 0; r < reps; ++r) {
	    scalStd(n, alpha, x);
	}
    });
    const double tFormula = seconds([&] {
	for (int r = 0; r < reps; ++r) {
	    scalFormula(n, alpha, x);
	}
    });
    std::printf("x = alpha x, n = %ld, %d times: operator* %.3f s, formula "
                "%.3f s\n",
                n, reps, tStd, tFormula);
}
