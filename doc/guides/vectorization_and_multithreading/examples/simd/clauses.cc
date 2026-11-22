// The optional clauses of omp simd, each in a small function.
#include <cstdio>
#include <cstdlib>
#include <vector>

// safelen(4): iteration i reads x[i] and writes x[i + 4].  Up to four
// consecutive iterations may run as one vector operation.
void
shiftAdd(long n, double *x)
{
#pragma omp simd safelen(4)
    for (long i = 0; i < n - 4; ++i) {
	x[i + 4] += x[i];
    }
}

// simdlen(8): a preferred number of lanes (here 8 floats).
void
square(long n, float *x)
{
#pragma omp simd simdlen(8)
    for (long i = 0; i < n; ++i) {
	x[i] *= x[i];
    }
}

// aligned(x, y : 64): both pointers are multiples of 64 bytes.
void
copyAligned(long n, const double *x, double *y)
{
#pragma omp simd aligned(x, y : 64)
    for (long i = 0; i < n; ++i) {
	y[i] = x[i];
    }
}

// linear(j : 2): j increases by 2 per iteration; without the clause the
// update of j would be a dependence between iterations.
void
gatherEven(long n, const double *x, double *y)
{
    long j = 0;
#pragma omp simd linear(j : 2)
    for (long i = 0; i < n; ++i) {
	y[i] = x[j];
	j += 2;
    }
}

// private(t): every lane has its own t.  lastprivate(last): after the
// loop last has the value of the last iteration.
double
privateLast(long n, const double *x, double *y)
{
    double t, last = 0;
#pragma omp simd private(t) lastprivate(last)
    for (long i = 0; i < n; ++i) {
	t = 2 * x[i];
	y[i] = t * t;
	last = y[i];
    }
    return last;
}

// collapse(2): the two loops form one iteration space of m*n iterations.
void
addMatrices(long m, long n, const double *A, double *B)
{
#pragma omp simd collapse(2)
    for (long i = 0; i < m; ++i) {
	for (long j = 0; j < n; ++j) {
	    B[i * n + j] += A[i * n + j];
	}
    }
}

// declare simd: the compiler also generates a vector version of the
// function, which a vectorized loop can call.
#pragma omp declare simd uniform(a) linear(i : 1) notinbranch
double
poly(double a, const double *x, long i)
{
    return (a * x[i] + 1) * x[i];
}

void
applyPoly(long n, double a, const double *x, double *y)
{
#pragma omp simd
    for (long i = 0; i < n; ++i) {
	y[i] = poly(a, x, i);
    }
}

// if(simd : n >= 64): only long loops are vectorized (OpenMP 5.0).
void
scaleIf(long n, double alpha, double *x)
{
#pragma omp simd if (simd : n >= 64)
    for (long i = 0; i < n; ++i) {
	x[i] *= alpha;
    }
}

int
main()
{
    const long n = 64;
    std::vector<double> x(n), y(n), z(2 * n);
    for (long i = 0; i < n; ++i) {
	x[i] = double(i);
    }
    for (long i = 0; i < 2 * n; ++i) {
	z[i] = double(i);
    }

    std::vector<double> s(12, 1.0);
    shiftAdd(12, s.data()); // 1 1 1 1 2 2 2 2 3 3 3 3
    std::printf("shiftAdd:    ");
    for (double v : s) {
	std::printf("%g ", v);
    }
    std::printf("\n");

    std::vector<float> f(16, 3.0f);
    square(16, f.data());
    std::printf("square:      %g\n", f[15]);

    double *a =
        static_cast<double *>(std::aligned_alloc(64, n * sizeof(double)));
    double *b =
        static_cast<double *>(std::aligned_alloc(64, n * sizeof(double)));
    for (long i = 0; i < n; ++i) {
	a[i] = double(i);
    }
    copyAligned(n, a, b);
    std::printf("copyAligned: %g\n", b[n - 1]);
    std::free(a);
    std::free(b);

    gatherEven(n, z.data(), y.data());
    std::printf("gatherEven:  %g %g %g\n", y[0], y[1], y[n - 1]);

    std::printf("privateLast: %g\n", privateLast(n, x.data(), y.data()));

    std::vector<double> A(6, 1.0), B(6, 2.0);
    addMatrices(2, 3, A.data(), B.data());
    std::printf("addMatrices: %g\n", B[5]);

    applyPoly(n, 2.0, x.data(), y.data());
    std::printf("applyPoly:   %g\n", y[3]); // (2*3+1)*3 = 21

    scaleIf(n, 0.5, x.data());
    std::printf("scaleIf:     %g\n", x[n - 1]);
}
