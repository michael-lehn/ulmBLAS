// LU with partial pivoting four ways: unblocked (getf2: level-2 BLAS),
// blocked right-looking with a fixed block size (getf2 for the panel,
// trsm and gemm for the rest), recursive (getrf2), and getrf, which also
// chooses a parallel variant.  Run with ULMBLAS_NUM_THREADS=1 for one
// thread.
#include "../common/util.h"
#include <ulmblas/lapack/getf2.h>
#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/getrf2.h>
#include <vector>

// blocked, right-looking (variant 5), block size nb, column major
long
getrfBlocked(long n, double *A, long *piv, long nb)
{
    long info = 0;
    for (long j = 0; j < n; j += nb) {
	const long b = std::min(nb, n - j);
	double *A11 = &A[j + j * n], *A12 = &A[j + (j + b) * n];
	double *A21 = &A[(j + b) + j * n], *A22 = &A[(j + b) + (j + b) * n];

	// the panel (A11; A21): unblocked, pivots relative to row j
	const long i = ulmBLAS::getf2(n - j, b, A11, 1L, n, &piv[j], 1L);
	if (i && !info) {
	    info = i + j;
	}
	for (long l = j; l < j + b; ++l) {
	    piv[l] += j;
	}
	// its interchanges left and right of the panel
	ulmBLAS::laswp(j, A, 1L, n, j, j + b, piv, 1L);
	ulmBLAS::laswp(n - j - b, &A[(j + b) * n], 1L, n, j, j + b, piv, 1L);
	// A12 := L11^{-1} A12 (trsm), A22 := A22 - A21 A12 (gemm)
	ulmBLAS::trlsm(b, n - j - b, 1.0, false, true, A11, 1L, n, A12, 1L, n);
	ulmBLAS::gemm(n - j - b, n - j - b, b, -1.0, false, A21, 1L, n, false,
	              A12, 1L, n, 1.0, A22, 1L, n);
    }
    return info;
}

int
main()
{
    std::printf("%6s %10s %10s %10s %10s   (GFLOPS)\n", "n", "unblocked",
                "nb = 64", "recursive", "getrf");
    for (long n : {256L, 512L, 1024L, 2048L, 4096L}) {
	std::vector<double> A0(n * n), A(n * n);
	std::vector<long> piv(n);
	randomMatrix(n, n, A0.data(), n);
	const double gflop = 2e-9 * n * n * n / 3;
	double g[4];
	for (int v = 0; v < 4; ++v) {
	    if (v == 0 && n > 2048) { // too slow
		g[v] = 0;
		continue;
	    }
	    g[v] = 0;
	    for (int run = 0; run < 3; ++run) { // the best of three runs
		A = A0;
		lap();
		if (v == 0) {
		    ulmBLAS::getf2(n, n, A.data(), 1L, n, piv.data(), 1L);
		} else if (v == 1) {
		    getrfBlocked(n, A.data(), piv.data(), 64);
		} else if (v == 2) {
		    ulmBLAS::getrf2(n, n, A.data(), 1L, n, piv.data(), 1L);
		} else {
		    ulmBLAS::getrf(n, n, A.data(), 1L, n, piv.data(), 1L);
		}
		g[v] = std::max(g[v], gflop / lap());
	    }
	    std::vector<double> B = A0; // check the result
	    if (luResidual(n, B.data(), A.data(), piv.data()) > 10) {
		std::printf("variant %d: wrong result\n", v);
	    }
	}
	std::printf("%6ld %10.1f %10.1f %10.1f %10.1f\n", n, g[0], g[1], g[2],
	            g[3]);
    }
}
