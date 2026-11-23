// syrk as gemm restricted to a triangle: C <- beta C + alpha A A^T for the
// lower triangle of the n x n matrix C, A n x k.  The frame of blocked/gemm.h
// with B = A^T (strides swapped), blocks of C above the diagonal skipped, and
// a macro kernel that skips the tiles above the diagonal and stores only the
// lower part of the tiles that cross it.  The upper triangle of C is not
// accessed.  Compared with gemm (the full C) and with ulmBLAS.
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

// the macro kernel for the lower triangle: the mc x nc block of C starts at
// row i0 and column j0 of the whole matrix; element (r,c) of the block is in
// the lower triangle if i0 + r >= j0 + c
template <typename UK>
void
macroKernelLower(long i0, long j0, long mc, long nc, long kc, double alpha,
                 const double *A_, const double *B_, double beta, double *C,
                 long incRowC, long incColC)
{
    const long MR = UK::MR, NR = UK::NR;
    alignas(64) double AB[MR * NR];
    for (long q = 0; q < (nc + NR - 1) / NR; ++q) {
	const long nr = std::min(NR, nc - q * NR);
	const long c0 = j0 + q * NR; // first column of the tile
	for (long p = 0; p < (mc + MR - 1) / MR; ++p) {
	    const long mr = std::min(MR, mc - p * MR);
	    const long r0 = i0 + p * MR; // first row of the tile
	    if (r0 + mr - 1 < c0) {
		continue; // above the diagonal
	    }
	    UK::run(kc, alpha, &A_[p * MR * kc], &B_[q * NR * kc], AB);
	    double *Cpq = &C[p * MR * incRowC + q * NR * incColC];
	    if (r0 >= c0 + nr - 1) { // below the diagonal
		blocked::update<MR, NR>(mr, nr, beta, AB, Cpq, incRowC,
		                        incColC);
		continue;
	    }
	    for (long j = 0; j < nr; ++j) { // crosses the diagonal
		for (long i = std::max(0L, c0 + j - r0); i < mr; ++i) {
		    double &c = Cpq[i * incRowC + j * incColC];
		    c = (beta == 0 ? 0 : beta * c) + AB[i + j * MR];
		}
	    }
	}
    }
}

// the frame (alpha != 0, n, k > 0)
template <typename UK, long MC, long KC, long NC>
void
syrk(long n, long k, double alpha, const double *A, long incRowA, long incColA,
     double beta, double *C, long incRowC, long incColC)
{
    alignas(64) static double A_[MC * KC];
    alignas(64) static double B_[KC * NC];

    for (long jc = 0; jc < n; jc += NC) {
	const long nc = std::min(NC, n - jc);
	for (long pc = 0; pc < k; pc += KC) {
	    const long kc = std::min(KC, k - pc);
	    const double b = pc == 0 ? beta : 1;
	    // B = A^T: rows pc..pc+kc-1, columns jc..jc+nc-1 of A^T
	    blocked::packB<UK::NR>(kc, nc, &A[pc * incColA + jc * incRowA],
	                           incColA, incRowA, B_);
	    for (long ic = 0; ic < n; ic += MC) {
		const long mc = std::min(MC, n - ic);
		if (ic + mc - 1 < jc) {
		    continue; // the block is above the diagonal
		}
		blocked::packA<UK::MR>(mc, kc, &A[ic * incRowA + pc * incColA],
		                       incRowA, incColA, A_);
		macroKernelLower<UK>(ic, jc, mc, nc, kc, alpha, A_, B_, b,
		                     &C[ic * incRowC + jc * incColC], incRowC,
		                     incColC);
	    }
	}
    }
}

// max |C - R| / max(|R|, 1) in the lower triangle; infinity if the upper
// triangle of C (NaN) was changed or the lower one has a NaN
double
differenceLower(long n, const double *C, const double *R)
{
    double d = 0, r = 1;
    for (long j = 0; j < n; ++j) {
	for (long i = 0; i < n; ++i) {
	    if (i < j) {
		if (!std::isnan(C[i + j * n])) {
		    return INFINITY;
		}
		continue;
	    }
	    const double e = std::abs(C[i + j * n] - R[i + j * n]);
	    if (std::isnan(e)) {
		return INFINITY;
	    }
	    d = std::max(d, e);
	    r = std::max(r, std::abs(R[i + j * n]));
	}
    }
    return d / r;
}

int
main()
{
    std::printf("%6s %10s %10s %10s %10s   (seconds)\n", "n = k", "gemm",
                "syrk", "ulm gemm", "ulm syrk");
    for (long n : {500L, 1000L, 2000L, 4000L}) {
	std::vector<double> A(n * n), C(n * n), R(n * n);
	randomMatrix(n, n, A.data(), 1, n, 1);
	ulmBLAS::gemm(n, n, n, 1.0, false, A.data(), 1L, n, false, A.data(), n,
	              1L, 0.0, R.data(), 1L, n); // reference result: A A^T
	std::printf("%6ld", n);
	for (int v = 0; v < 4; ++v) {
	    double best = INFINITY;
	    for (int run = 0; run < 2; ++run) {
		for (long j = 0; j < n; ++j) { // upper triangle: NaN
		    for (long i = 0; i < n; ++i) {
			C[i + j * n] = i < j && v % 2 == 1 ? std::nan("") : 0;
		    }
		}
		lap();
		if (v == 0) {
		    blocked::gemm<UK, 96, 256, 4104>(n, n, n, 1.0, A.data(), 1,
		                                     n, A.data(), n, 1, 0.0,
		                                     C.data(), 1, n);
		} else if (v == 1) {
		    syrk<UK, 96, 256, 4104>(n, n, 1.0, A.data(), 1, n, 0.0,
		                            C.data(), 1, n);
		} else if (v == 2) {
		    ulmBLAS::gemm(n, n, n, 1.0, false, A.data(), 1L, n, false,
		                  A.data(), n, 1L, 0.0, C.data(), 1L, n);
		} else {
		    ulmBLAS::sylrk(n, n, 1.0, A.data(), 1L, n, 0.0, C.data(),
		                   1L, n);
		}
		best = std::min(best, lap());
	    }
	    const double d = v % 2 == 1
	                         ? differenceLower(n, C.data(), R.data())
	                         : difference(n, n, C.data(), R.data(), n);
	    if (d > 1e-10) {
		std::printf(" %10s", "WRONG");
	    } else {
		std::printf(" %10.3f", best);
	    }
	}
	std::printf("\n");
    }
}
