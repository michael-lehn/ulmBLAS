// x <- alpha*x in blocks of 32 elements, with the pragma in a macro
// (as ULMBLAS_SIMD in ulmblas/level1/kernel/simd.h).
#include <cstdio>
#include <vector>

#define PRAGMA(x) _Pragma(#x)
#define SIMD PRAGMA(omp simd)

void
scal(long n, double alpha, double *x)
{
    const int B = 32;
    long i = 0;
    for (; i + B <= n; i += B) {
	SIMD for (int k = 0; k < B; ++k)
	{
	    x[i + k] *= alpha;
	}
    }
    for (; i < n; ++i) { // the remaining n mod 32 elements
	x[i] *= alpha;
    }
}

int
main()
{
    const long n = 100;
    std::vector<double> x(n, 1.0);
    scal(n, 3.0, x.data());
    long wrong = 0;
    for (double v : x) {
	wrong += v != 3.0;
    }
    std::printf("n = %ld, wrong elements: %ld\n", n, wrong);
}
