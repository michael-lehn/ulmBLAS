// A complete gemm in the structure of BLIS and ulmBLAS: C <- beta C + alpha
// A B with the five loops of the frame algorithm, packing of A and B, a
// macro kernel and an exchangeable micro kernel.  All matrices have
// arbitrary strides: X(i,j) = X[i*incRowX + j*incColX].
//
// Parameters: a micro kernel UK (a struct with MR, NR and a static
// function), and the block sizes MC, KC, NC.
#ifndef GEMM_H
#define GEMM_H 1

#include <algorithm>
#include <cstdlib>

namespace blocked {

// pack the mc x kc block of A into panels of MR rows: element (i,l) of
// panel p at A_[p*MR*kc + l*MR + i]; the last panel is filled with zeros
template <long MR>
void
packA(long mc, long kc, const double *A, long incRowA, long incColA, double *A_)
{
    for (long p = 0; p < (mc + MR - 1) / MR; ++p) {
	const long mr = std::min(MR, mc - p * MR);
	for (long l = 0; l < kc; ++l) {
	    for (long i = 0; i < MR; ++i) {
		A_[i] = i < mr ? A[(p * MR + i) * incRowA + l * incColA] : 0;
	    }
	    A_ += MR;
	}
    }
}

// pack the kc x nc block of B into panels of NR columns: element (l,j) of
// panel q at B_[q*NR*kc + l*NR + j]
template <long NR>
void
packB(long kc, long nc, const double *B, long incRowB, long incColB, double *B_)
{
    for (long q = 0; q < (nc + NR - 1) / NR; ++q) {
	const long nr = std::min(NR, nc - q * NR);
	for (long l = 0; l < kc; ++l) {
	    for (long j = 0; j < NR; ++j) {
		B_[j] = j < nr ? B[l * incRowB + (q * NR + j) * incColB] : 0;
	    }
	    B_ += NR;
	}
    }
}

// C <- beta C + AB for an mr x nr tile, AB computed by the micro kernel
// into a buffer (column major, MR x NR)
template <long MR, long NR>
void
update(long mr, long nr, double beta, const double *AB, double *C, long incRowC,
       long incColC)
{
    for (long j = 0; j < nr; ++j) {
	for (long i = 0; i < mr; ++i) {
	    double &c = C[i * incRowC + j * incColC];
	    c = (beta == 0 ? 0 : beta * c) + AB[i + j * MR];
	}
    }
}

// the macro kernel: C <- beta C + alpha A_ B_ for an mc x nc block of C,
// tile by tile; never inlined, so that it is compiled on its own
template <typename UK>
__attribute__((noinline)) void
macroKernel(long mc, long nc, long kc, double alpha, const double *A_,
            const double *B_, double beta, double *C, long incRowC,
            long incColC)
{
    const long MR = UK::MR, NR = UK::NR;
    alignas(64) double AB[MR * NR];
    for (long q = 0; q < (nc + NR - 1) / NR; ++q) {
	const long nr = std::min(NR, nc - q * NR);
	for (long p = 0; p < (mc + MR - 1) / MR; ++p) {
	    const long mr = std::min(MR, mc - p * MR);
	    UK::run(kc, alpha, &A_[p * MR * kc], &B_[q * NR * kc], AB);
	    update<MR, NR>(mr, nr, beta, AB,
	                   &C[p * MR * incRowC + q * NR * incColC], incRowC,
	                   incColC);
	}
    }
}

// the frame algorithm
template <typename UK, long MC, long KC, long NC>
void
gemm(long m, long n, long k, double alpha, const double *A, long incRowA,
     long incColA, const double *B, long incRowB, long incColB, double beta,
     double *C, long incRowC, long incColC)
{
    static_assert(MC % UK::MR == 0 && NC % UK::NR == 0, "whole panels");
    alignas(64) static double A_[MC * KC];
    alignas(64) static double B_[KC * NC];

    if (k == 0 || alpha == 0) { // C <- beta C
	for (long j = 0; j < n; ++j) {
	    for (long i = 0; i < m; ++i) {
		double &c = C[i * incRowC + j * incColC];
		c = beta == 0 ? 0 : beta * c;
	    }
	}
	return;
    }
    for (long jc = 0; jc < n; jc += NC) { // loop 5: blocks of columns
	const long nc = std::min(NC, n - jc);
	for (long pc = 0; pc < k; pc += KC) { // loop 4: rank-KC updates
	    const long kc = std::min(KC, k - pc);
	    const double b = pc == 0 ? beta : 1; // beta only once
	    packB<UK::NR>(kc, nc, &B[pc * incRowB + jc * incColB], incRowB,
	                  incColB, B_);
	    for (long ic = 0; ic < m; ic += MC) { // loop 3: blocks of rows
		const long mc = std::min(MC, m - ic);
		packA<UK::MR>(mc, kc, &A[ic * incRowA + pc * incColA], incRowA,
		              incColA, A_);
		macroKernel<UK>(mc, nc, kc, alpha, A_, B_, b,
		                &C[ic * incRowC + jc * incColC], incRowC,
		                incColC);
	    }
	}
    }
}

} // namespace blocked

#endif // GEMM_H
