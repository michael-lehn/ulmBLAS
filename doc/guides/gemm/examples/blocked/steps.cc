// From the naive loop to the structure of BLIS: the same blocked gemm with
// three micro kernels (the third with NEON or AVX2 intrinsics), compared with a
// naive loop and with ulmBLAS::gemm. One thread; C = A B, all n x n, column
// major.
#include "../common/util.h"
#include "gemm.h"
#include "kernels.h"
#include <ulmblas/ulmblas.h>
#include <vector>

void
naive(long n, const double *A, const double *B, double *C)
{
    for (long j = 0; j < n; ++j) { // jli: i innermost, along columns
	for (long l = 0; l < n; ++l) {
	    for (long i = 0; i < n; ++i) {
		C[i + j * n] += A[i + l * n] * B[l + j * n];
	    }
	}
    }
}

typedef void (*Gemm)(long, long, long, double, const double *, long, long,
                     const double *, long, long, double, double *, long, long);

int
main()
{
    const char *names[] = {"naive", "ref 4x4", "vec 8x6", "simd 8x6",
                           "ulmBLAS"};
    Gemm blockedGemm[] = {
        nullptr, blocked::gemm<blocked::RefKernel<4, 4>, 96, 256, 4104>,
        blocked::gemm<blocked::VecKernel, 96, 256, 4104>,
#if defined(HAVE_SIMD_KERNEL)
        blocked::gemm<blocked::SimdKernel, 96, 256, 4104>,
#else
        nullptr,
#endif
        nullptr};

    std::printf("%6s", "n");
    for (const char *name : names) {
	std::printf(" %9s", name);
    }
    std::printf("   (GFLOPS)\n");
    for (long n : {256L, 512L, 1024L, 2048L, 4096L}) {
	std::vector<double> A(n * n), B(n * n), C(n * n), R(n * n);
	randomMatrix(n, n, A.data(), 1, n, 1);
	randomMatrix(n, n, B.data(), 1, n, 2);
	ulmBLAS::gemm(n, n, n, 1.0, false, A.data(), 1L, n, false, B.data(), 1L,
	              n, 0.0, R.data(), 1L, n); // reference result
	std::printf("%6ld", n);
	for (int v = 0; v < 5; ++v) {
	    if ((v == 0 && n > 2048) || (v == 3 && !blockedGemm[3])) {
		std::printf(" %9s", "-");
		continue;
	    }
	    double best = 0;
	    for (int run = 0; run < 2; ++run) {
		std::fill(C.begin(), C.end(), 0.0);
		lap();
		if (v == 0) {
		    naive(n, A.data(), B.data(), C.data());
		} else if (v == 4) {
		    ulmBLAS::gemm(n, n, n, 1.0, false, A.data(), 1L, n, false,
		                  B.data(), 1L, n, 0.0, C.data(), 1L, n);
		} else {
		    blockedGemm[v](n, n, n, 1.0, A.data(), 1, n, B.data(), 1, n,
		                   0.0, C.data(), 1, n);
		}
		best = std::max(best, 2e-9 * n * n * n / lap());
	    }
	    if (difference(n, n, C.data(), R.data(), n) > 1e-10) {
		std::printf(" %9s", "WRONG");
	    } else {
		std::printf(" %9.1f", best);
	    }
	}
	std::printf("\n");
    }
}
