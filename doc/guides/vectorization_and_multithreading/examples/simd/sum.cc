// The sum of a vector: without and with a reduction clause.
#include <chrono>
#include <cstdio>
#include <vector>

float
sumPlain(long n, const float *x)
{
    float s = 0;
    for (long i = 0; i < n; ++i) {
	s += x[i];
    }
    return s;
}

float
sumSimd(long n, const float *x)
{
    float s = 0;
#pragma omp simd reduction(+ : s)
    for (long i = 0; i < n; ++i) {
	s += x[i];
    }
    return s;
}

template <typename F>
double
nsPerCall(F f, int reps)
{
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
	// the compiler must assume that memory has changed: no hoisting
	asm volatile("" : : : "memory");
	f();
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
}

int
main()
{
    const long n = 4000;
    std::vector<float> x(n);
    for (long i = 0; i < n; ++i) {
	x[i] = 1.0f / float(i + 1);
    }
    volatile float sink;
    double tPlain = nsPerCall([&] { sink = sumPlain(n, x.data()); }, 100000);
    double tSimd = nsPerCall([&] { sink = sumSimd(n, x.data()); }, 100000);
    (void)sink;
    std::printf("plain: %.9g  %8.1f ns\n", sumPlain(n, x.data()), tPlain);
    std::printf("simd:  %.9g  %8.1f ns\n", sumSimd(n, x.data()), tSimd);
}
