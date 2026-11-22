// A target for a profiler: the recursive LU or Cholesky factorization of an
// n x n matrix, repeated, on the threads that ulmBLAS uses (set
// ULMBLAS_NUM_THREADS=1 for one).  Usage: factor lu|chol [n [repetitions]]
#include "../common/util.h"
#include <cstdlib>
#include <cstring>
#include <ulmblas/lapack/getrf2.h>
#include <ulmblas/lapack/potrf2.h>
#include <vector>

int
main(int argc, char **argv)
{
    const bool lu = argc < 2 || std::strcmp(argv[1], "chol") != 0;
    const long n = argc > 2 ? std::atol(argv[2]) : 2000;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20;

    std::vector<double> A0(n * n), A(n * n), W(n * n);
    std::vector<long> piv(n);
    if (lu) {
	randomMatrix(n, n, A0.data(), n);
    } else {
	spdMatrix(n, A0.data(), W.data());
    }
    double t = 0;
    for (int r = 0; r < reps; ++r) {
	A = A0;
	lap();
	if (lu) {
	    ulmBLAS::getrf2(n, n, A.data(), 1L, n, piv.data(), 1L);
	} else {
	    ulmBLAS::potrf2(n, A.data(), 1L, n);
	}
	t += lap();
    }
    const double flops = (lu ? 2.0 : 1.0) * n * n * n / 3;
    std::printf("%s, n = %ld: %.1f GFLOPS\n", lu ? "getrf2" : "potrf2", n,
                1e-9 * flops * reps / t);
}
