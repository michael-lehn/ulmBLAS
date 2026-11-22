// The Cholesky factorization unblocked (potf2) and recursive (potrf2, the
// algorithm of potrf without OpenMP).  Run with ULMBLAS_NUM_THREADS=1 for
// one thread.
#include "../common/util.h"
#include <ulmblas/lapack/potf2.h>
#include <ulmblas/lapack/potrf2.h>
#include <vector>

int
main()
{
    std::printf("%6s %10s %10s   (GFLOPS)\n", "n", "unblocked", "recursive");
    for (long n : {256L, 512L, 1024L, 2048L, 4096L}) {
	std::vector<double> A0(n * n), W(n * n), A(n * n);
	spdMatrix(n, A0.data(), W.data());
	const double gflop = 1e-9 * n * n * n / 3;
	double g[2] = {0, 0};
	for (int v = 0; v < 2; ++v) {
	    for (int run = 0; run < 3; ++run) { // the best of three runs
		A = A0;
		lap();
		const long info = v == 0 ? ulmBLAS::potf2(n, A.data(), 1L, n)
		                         : ulmBLAS::potrf2(n, A.data(), 1L, n);
		g[v] = std::max(g[v], gflop / lap());
		if (info) {
		    std::printf("info = %ld\n", info);
		}
	    }
	}
	std::printf("%6ld %10.1f %10.1f\n", n, g[0], g[1]);
    }
}
