// Helpers of the examples: column-major matrices with strides
// (A(i,j) = A[i*incRow + j*incCol]), random numbers, a timer, and the
// difference of two results.
#ifndef UTIL_H
#define UTIL_H 1

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

// a reproducible random number in [-1, 1)
inline double
random(unsigned long &state)
{
    state = state * 6364136223846793005UL + 1442695040888963407UL;
    return double(state >> 11) / double(1UL << 52) - 1;
}

inline void
randomMatrix(long m, long n, double *A, long incRow, long incCol,
             unsigned long seed = 1)
{
    for (long j = 0; j < n; ++j) {
	for (long i = 0; i < m; ++i) {
	    A[i * incRow + j * incCol] = random(seed);
	}
    }
}

// max |A - B| / max(|B|, 1); infinity if A has a NaN
inline double
difference(long m, long n, const double *A, const double *B, long ld)
{
    double d = 0, b = 1;
    for (long j = 0; j < n; ++j) {
	for (long i = 0; i < m; ++i) {
	    const double e = std::abs(A[i + j * ld] - B[i + j * ld]);
	    if (std::isnan(e)) {
		return INFINITY; // a NaN in the result is an error, too
	    }
	    d = std::max(d, e);
	    b = std::max(b, std::abs(B[i + j * ld]));
	}
    }
    return d / b;
}

// seconds since the last call
inline double
lap()
{
    static auto t0 = std::chrono::steady_clock::now();
    const auto t1 = std::chrono::steady_clock::now();
    const double t = std::chrono::duration<double>(t1 - t0).count();
    t0 = t1;
    return t;
}

#endif // UTIL_H
