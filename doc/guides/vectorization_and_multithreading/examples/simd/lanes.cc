// Dot product three ways: plain loop, omp simd reduction, and explicit
// lanes (partial sums) as in ulmblas/level1/kernel/reduce.h.
#include <chrono>
#include <cstdio>
#include <vector>

double
dotPlain(long n, const double *x, const double *y)
{
    double s = 0;
    for (long i = 0; i < n; ++i) {
	s += x[i] * y[i];
    }
    return s;
}

double
dotReduction(long n, const double *x, const double *y)
{
    double s = 0;
#pragma omp simd reduction(+ : s)
    for (long i = 0; i < n; ++i) {
	s += x[i] * y[i];
    }
    return s;
}

// K partial sums: 128 bytes of accumulators (16 doubles).  The loop over
// the lanes is the vectorized loop; the K additions of a block are
// independent, so they hide the latency of the floating-point adder.
double
dotLanes(long n, const double *x, const double *y)
{
    const int K = 128 / sizeof(double);
    double lane[K] = {};
    long i = 0;
    for (; i + K <= n; i += K) {
#pragma omp simd
	for (int k = 0; k < K; ++k) {
	    lane[k] += x[i + k] * y[i + k];
	}
    }
    // combine the lanes pairwise: K/2, K/4, ..., 1
    for (int w = K / 2; w >= 1; w /= 2) {
#pragma omp simd
	for (int k = 0; k < w; ++k) {
	    lane[k] += lane[k + w];
	}
    }
    double s = lane[0];
    for (; i < n; ++i) {
	s += x[i] * y[i];
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
    const long n = 1000;
    std::vector<double> x(n), y(n);
    for (long i = 0; i < n; ++i) {
	x[i] = 1.0 / double(i + 1);
	y[i] = double(i % 7) - 3;
    }
    volatile double sink;
    const int reps = 1000000;
    auto run = [&](const char *name,
                   double (*dot)(long, const double *, const double *)) {
	double t = nsPerCall([&] { sink = dot(n, x.data(), y.data()); }, reps);
	std::printf("%-10s %.17g %7.1f ns  %5.2f elements/ns\n", name,
	            dot(n, x.data(), y.data()), t, n / t);
    };
    run("plain", dotPlain);
    run("reduction", dotReduction);
    run("lanes", dotLanes);
    (void)sink;
}
