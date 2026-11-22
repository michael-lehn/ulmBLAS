// Helpers of the examples: matrices in column-major order (A(i,j) at
// A[i + j*ld]), random numbers, a timer, and the residual of an LU or
// Cholesky factorization.
#ifndef UTIL_H
#define UTIL_H 1

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <ulmblas/lapack/laswp.h>
#include <ulmblas/ulmblas.h>

// a reproducible random number in [-1, 1)
inline double
random(unsigned long &state)
{
    state = state * 6364136223846793005UL + 1442695040888963407UL;
    return double(state >> 11) / double(1UL << 52) - 1;
}

inline void
randomMatrix(long m, long n, double *A, long ld, unsigned long seed = 1)
{
    for (long j = 0; j < n; ++j) {
	for (long i = 0; i < m; ++i) {
	    A[i + j * ld] = random(seed);
	}
    }
}

// a symmetric positive definite matrix: B B^T + n I, B random
inline void
spdMatrix(long n, double *A, double *work, unsigned long seed = 1)
{
    randomMatrix(n, n, work, n, seed);
    ulmBLAS::gemm(n, n, n, 1.0, false, work, 1L, n, false, work, n, 1L, 0.0,
                  A, 1L, n);
    for (long i = 0; i < n; ++i) {
	A[i + i * n] += double(n);
    }
}

inline void
print(const char *name, long m, long n, const double *A, long ld)
{
    std::printf("%s =\n", name);
    for (long i = 0; i < m; ++i) {
	for (long j = 0; j < n; ++j) {
	    std::printf(" %8.4f", A[i + j * ld]);
	}
	std::printf("\n");
    }
}

inline double
normMax(long m, long n, const double *A, long ld)
{
    double s = 0;
    for (long j = 0; j < n; ++j) {
	for (long i = 0; i < m; ++i) {
	    s = std::max(s, std::abs(A[i + j * ld]));
	}
    }
    return s;
}

// ||P A - L U|| / (n eps ||A||) for the result LU of getrf (n x n): L and U
// are copied out of LU, A0 is the original matrix (overwritten)
inline double
luResidual(long n, double *A0, const double *LU, const long *piv)
{
    const double nrmA = normMax(n, n, A0, n);
    double *L = new double[n * n], *U = new double[n * n];
    for (long j = 0; j < n; ++j) {
	for (long i = 0; i < n; ++i) {
	    const double a = LU[i + j * n];
	    L[i + j * n] = i > j ? a : (i == j ? 1 : 0);
	    U[i + j * n] = i <= j ? a : 0;
	}
    }
    ulmBLAS::laswp(n, A0, 1L, n, 0L, n, piv, 1L); // A0 := P A0
    ulmBLAS::gemm(n, n, n, -1.0, false, L, 1L, n, false, U, 1L, n, 1.0, A0,
                  1L, n);
    delete[] L;
    delete[] U;
    const double eps = std::numeric_limits<double>::epsilon();
    return normMax(n, n, A0, n) / (n * eps * nrmA);
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
