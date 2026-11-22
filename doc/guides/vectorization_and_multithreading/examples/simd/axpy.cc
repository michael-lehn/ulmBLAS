// y <- alpha*x + y, once as a plain loop, once with omp simd.
// Compile with vectorization remarks, for example:
//   clang++ -O2 -fopenmp-simd -Rpass=loop-vectorize -c axpy.cc
// (-O2: optimization level 2)
#include <cstddef>

void
axpyPlain(std::ptrdiff_t n, double alpha, const double *x, double *y)
{
    for (std::ptrdiff_t i = 0; i < n; ++i) {
	y[i] += alpha * x[i];
    }
}

void
axpySimd(std::ptrdiff_t n, double alpha, const double *x, double *y)
{
#pragma omp simd
    for (std::ptrdiff_t i = 0; i < n; ++i) {
	y[i] += alpha * x[i];
    }
}

// With a stride the compiler has to gather and scatter; omp simd
// still asserts that the iterations are independent.
void
axpyStrided(std::ptrdiff_t n, double alpha, const double *x,
            std::ptrdiff_t incX, double *y, std::ptrdiff_t incY)
{
#pragma omp simd
    for (std::ptrdiff_t i = 0; i < n; ++i) {
	y[i * incY] += alpha * x[i * incX];
    }
}
