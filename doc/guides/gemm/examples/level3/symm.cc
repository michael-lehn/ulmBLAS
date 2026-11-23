// symm as gemm with another packing: C <- beta C + alpha S B for a symmetric
// m x m matrix S of which only the lower triangle is read.  The frame and the
// macro kernel are those of blocked/gemm.h; only the packing of S differs.
// Compared with the blocked gemm of the full matrix and with ulmBLAS.
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

// element (i,l) of S from its lower triangle
inline double
sym(const double *S, long incRowS, long incColS, long i, long l)
{
    return i >= l ? S[i * incRowS + l * incColS] : S[l * incRowS + i * incColS];
}

// as blocked::packA, for the mc x kc block of S at row i0, column l0
template <long MR>
void
packSym(long i0, long l0, long mc, long kc, const double *S, long incRowS,
        long incColS, double *A_)
{
    for (long p = 0; p < (mc + MR - 1) / MR; ++p) {
	const long mr = std::min(MR, mc - p * MR);
	for (long l = 0; l < kc; ++l) {
	    for (long i = 0; i < MR; ++i) {
		A_[i] = i < mr
		            ? sym(S, incRowS, incColS, i0 + p * MR + i, l0 + l)
		            : 0;
	    }
	    A_ += MR;
	}
    }
}

// the frame of blocked::gemm with packSym for A = S (alpha != 0, m, n > 0)
template <typename UK, long MC, long KC, long NC>
void
symm(long m, long n, double alpha, const double *S, long incRowS, long incColS,
     const double *B, long incRowB, long incColB, double beta, double *C,
     long incRowC, long incColC)
{
    alignas(64) static double A_[MC * KC];
    alignas(64) static double B_[KC * NC];

    for (long jc = 0; jc < n; jc += NC) {
	const long nc = std::min(NC, n - jc);
	for (long pc = 0; pc < m; pc += KC) {
	    const long kc = std::min(KC, m - pc);
	    const double b = pc == 0 ? beta : 1;
	    blocked::packB<UK::NR>(kc, nc, &B[pc * incRowB + jc * incColB],
	                           incRowB, incColB, B_);
	    for (long ic = 0; ic < m; ic += MC) {
		const long mc = std::min(MC, m - ic);
		packSym<UK::MR>(ic, pc, mc, kc, S, incRowS, incColS, A_);
		blocked::macroKernel<UK>(mc, nc, kc, alpha, A_, B_, b,
		                         &C[ic * incRowC + jc * incColC],
		                         incRowC, incColC);
	    }
	}
    }
}

int
main()
{
    std::printf("%6s %10s %10s %10s %10s   (GFLOPS)\n", "n", "gemm", "symm",
                "ulm gemm", "ulm symm");
    for (long n : {500L, 1000L, 2000L, 4000L}) {
	std::vector<double> S(n * n), F(n * n), B(n * n), C(n * n), R(n * n);
	// S: the lower triangle random, the upper one NaN (never read);
	// F: the full symmetric matrix for gemm
	randomMatrix(n, n, S.data(), 1, n, 1);
	for (long j = 0; j < n; ++j) {
	    for (long i = 0; i < n; ++i) {
		F[i + j * n] = sym(S.data(), 1, n, i, j);
		if (i < j) {
		    S[i + j * n] = std::nan("");
		}
	    }
	}
	randomMatrix(n, n, B.data(), 1, n, 2);
	ulmBLAS::gemm(n, n, n, 1.0, false, F.data(), 1L, n, false, B.data(), 1L,
	              n, 0.0, R.data(), 1L, n); // reference result
	std::printf("%6ld", n);
	for (int v = 0; v < 4; ++v) {
	    double best = 0;
	    for (int run = 0; run < 2; ++run) {
		lap();
		if (v == 0) {
		    blocked::gemm<UK, 96, 256, 4104>(n, n, n, 1.0, F.data(), 1,
		                                     n, B.data(), 1, n, 0.0,
		                                     C.data(), 1, n);
		} else if (v == 1) {
		    symm<UK, 96, 256, 4104>(n, n, 1.0, S.data(), 1, n, B.data(),
		                            1, n, 0.0, C.data(), 1, n);
		} else if (v == 2) {
		    ulmBLAS::gemm(n, n, n, 1.0, false, F.data(), 1L, n, false,
		                  B.data(), 1L, n, 0.0, C.data(), 1L, n);
		} else {
		    ulmBLAS::sylmm(n, n, 1.0, S.data(), 1L, n, B.data(), 1L, n,
		                   0.0, C.data(), 1L, n);
		}
		best = std::max(best, 2e-9 * n * n * n / lap());
	    }
	    if (difference(n, n, C.data(), R.data(), n) > 1e-10) {
		std::printf(" %10s", "WRONG");
	    } else {
		std::printf(" %10.1f", best);
	    }
	}
	std::printf("\n");
    }
}
