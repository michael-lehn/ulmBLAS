// trsm from gemm: B <- A^{-1} B for a lower triangular m x m matrix A (only
// its lower triangle is read), B m x n.  The diagonal blocks of KB rows are
// solved by substitution, the rows below them are updated by the blocked
// gemm of blocked/gemm.h.  For several KB: the speed and the share of the
// time spent in the substitution; for comparison ulmBLAS::trlsm.
#include "../blocked/gemm.h"
#include "../blocked/kernels.h"
#include "../common/util.h"
#include <ulmblas/ulmblas.h>
#include <vector>

#if defined(HAVE_SIMD_KERNEL)
typedef blocked::SimdKernel UK;
#else
typedef blocked::VecKernel UK;
#endif

// B <- A^{-1} B by substitution, column by column (A m x m, B m x n)
void
substitution(long m, long n, const double *A, long incRowA, long incColA,
             double *B, long incRowB, long incColB)
{
    for (long j = 0; j < n; ++j) {
	double *b = &B[j * incColB];
	for (long l = 0; l < m; ++l) {
	    b[l * incRowB] /= A[l * (incRowA + incColA)];
	    for (long i = l + 1; i < m; ++i) {
		b[i * incRowB] -= A[i * incRowA + l * incColA] * b[l * incRowB];
	    }
	}
    }
}

double timeSubstitution; // seconds spent in substitution

// the blocked algorithm: per diagonal block A11 (kb x kb)
//      B1 <- A11^{-1} B1           (substitution)
//      B2 <- B2 - A21 B1           (gemm)
void
trsm(long KB, long m, long n, const double *A, long incRowA, long incColA,
     double *B, long incRowB, long incColB)
{
    for (long l = 0; l < m; l += KB) {
	const long kb = std::min(KB, m - l);
	const auto t0 = std::chrono::steady_clock::now();
	substitution(kb, n, &A[l * (incRowA + incColA)], incRowA, incColA,
	             &B[l * incRowB], incRowB, incColB);
	timeSubstitution +=
	    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
	        .count();
	blocked::gemm<UK, 96, 256, 4104>(
	    m - l - kb, n, kb, -1.0, &A[(l + kb) * incRowA + l * incColA],
	    incRowA, incColA, &B[l * incRowB], incRowB, incColB, 1.0,
	    &B[(l + kb) * incRowB], incRowB, incColB);
    }
}

int
main()
{
    const long m = 2048, n = 2048;
    std::vector<double> A(m * m), B0(m * n), B(m * n), R(m * n);
    // A: lower triangle random, diagonal m (well conditioned), upper NaN
    randomMatrix(m, m, A.data(), 1, m, 1);
    for (long j = 0; j < m; ++j) {
	A[j + j * m] = m;
	for (long i = 0; i < j; ++i) {
	    A[i + j * m] = std::nan("");
	}
    }
    randomMatrix(m, n, B0.data(), 1, m, 2);
    R = B0;
    ulmBLAS::trlsm(m, n, 1.0, false, false, A.data(), 1L, m, R.data(), 1L, m);

    std::printf("m = n = %ld, one thread\n", m);
    std::printf("%8s %8s %14s\n", "KB", "GFLOPS", "substitution");
    for (long KB : {16L, 64L, 256L, 1024L, 2048L}) {
	double best = INFINITY, sub = 0;
	for (int run = 0; run < 2; ++run) {
	    B = B0;
	    timeSubstitution = 0;
	    lap();
	    trsm(KB, m, n, A.data(), 1, m, B.data(), 1, m);
	    const double t = lap();
	    if (t < best) {
		best = t;
		sub = timeSubstitution / t;
	    }
	}
	if (difference(m, n, B.data(), R.data(), m) > 1e-10) {
	    std::printf("%8ld %8s\n", KB, "WRONG");
	} else {
	    std::printf("%8ld %8.1f %13.0f%%\n", KB, 1e-9 * m * m * n / best,
	                100 * sub);
	}
    }
    double best = INFINITY;
    for (int run = 0; run < 2; ++run) {
	B = B0;
	lap();
	ulmBLAS::trlsm(m, n, 1.0, false, false, A.data(), 1L, m, B.data(), 1L,
	               m);
	best = std::min(best, lap());
    }
    std::printf("%8s %8.1f\n", "ulmBLAS", 1e-9 * m * m * n / best);
    best = INFINITY;
    for (int run = 0; run < 2; ++run) {
	lap();
	blocked::gemm<UK, 96, 256, 4104>(m, n, m, 1.0, R.data(), 1, m,
	                                 B0.data(), 1, m, 0.0, B.data(), 1, m);
	best = std::min(best, lap());
    }
    std::printf("%8s %8.1f   (the blocked gemm, %ld x %ld x %ld)\n", "gemm",
                2e-9 * m * m * n / best, m, n, m);
}
