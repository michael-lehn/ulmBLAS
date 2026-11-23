// The level-3 functions of ulmBLAS on one thread, n x n matrices: the time
// and the speed, each counted with its own number of operations.
#include "../common/util.h"
#include <ulmblas/ulmblas.h>
#include <vector>

int
main()
{
    const long n = 2000;
    std::vector<double> A(n * n), B(n * n), C(n * n), T(n * n);
    randomMatrix(n, n, A.data(), 1, n, 1);
    randomMatrix(n, n, B.data(), 1, n, 2);
    randomMatrix(n, n, T.data(), 1, n, 3);
    for (long j = 0; j < n; ++j) { // T lower triangular, well conditioned
	T[j + j * n] = n;
    }
    const double N = n;
    struct
    {
	const char *name;
	double ops; // operations
    } fn[] = {
        {"gemm    C <- A B", 2 * N * N * N},
        {"symm    C <- S B", 2 * N * N * N},
        {"syrk    C <- A A^T", N * (N + 1) * N},
        {"syr2k   C <- A B^T + B A^T", 2 * N * (N + 1) * N},
        {"gemmt   C <- A B (triangle)", N * (N + 1) * N},
        {"trmm    B <- T B", N * N * N},
        {"trsm    B <- T^{-1} B", N * N * N},
    };
    std::printf("n = %ld, one thread\n", n);
    std::printf("%-28s %8s %8s\n", "", "seconds", "GFLOPS");
    for (int v = 0; v < 7; ++v) {
	double best = INFINITY;
	for (int run = 0; run < 2; ++run) {
	    std::vector<double> X = B; // trmm and trsm overwrite their B
	    lap();
	    switch (v) {
	    case 0:
		ulmBLAS::gemm(n, n, n, 1.0, false, A.data(), 1L, n, false,
		              B.data(), 1L, n, 0.0, C.data(), 1L, n);
		break;
	    case 1:
		ulmBLAS::sylmm(n, n, 1.0, A.data(), 1L, n, B.data(), 1L, n, 0.0,
		               C.data(), 1L, n);
		break;
	    case 2:
		ulmBLAS::sylrk(n, n, 1.0, A.data(), 1L, n, 0.0, C.data(), 1L,
		               n);
		break;
	    case 3:
		ulmBLAS::sylr2k(n, n, 1.0, A.data(), 1L, n, B.data(), 1L, n,
		                0.0, C.data(), 1L, n);
		break;
	    case 4:
		ulmBLAS::gemmtr(true, n, n, 1.0, false, A.data(), 1L, n, false,
		                B.data(), 1L, n, 0.0, C.data(), 1L, n);
		break;
	    case 5:
		ulmBLAS::trlmm(n, n, 1.0, false, false, T.data(), 1L, n,
		               X.data(), 1L, n);
		break;
	    case 6:
		ulmBLAS::trlsm(n, n, 1.0, false, false, T.data(), 1L, n,
		               X.data(), 1L, n);
		break;
	    }
	    best = std::min(best, lap());
	}
	std::printf("%-28s %8.3f %8.1f\n", fn[v].name, best,
	            1e-9 * fn[v].ops / best);
    }
}
