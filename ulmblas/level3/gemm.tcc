#ifndef ULMBLAS_LEVEL3_GEMM_TCC
#define ULMBLAS_LEVEL3_GEMM_TCC 1

#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/fpsrscope.h>
#include <ulmblas/level3/gemm/select.h>
#include <ulmblas/level3/gemmdirect.h>
#include <ulmblas/level3/ukernel/ugemm.h>

//
//  gemm is a wrapper: it handles the special cases (alpha = 0 or k = 0,
//  the orientation of C, small matrices with gemm_direct) and calls the
//  frame algorithm (ulmblas/level3/gemm/frame.h) that the build selects for
//  the type of the packed blocks (ulmblas/level3/gemm/default.h).
//
namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
gemm(IndexType m, IndexType n, IndexType k, const Alpha &alpha, bool conjA,
     const TA *A, IndexType incRowA, IndexType incColA, bool conjB, const TB *B,
     IndexType incRowB, IndexType incColB, const Beta &beta, TC *C,
     IndexType incRowC, IndexType incColC)
{
    const FpsrScope fpsrScope; // see level3/fpsrscope.h

    typedef decltype(Alpha(0) * TA(0) * TB(0)) T;

    if (alpha == Alpha(0) || k == 0) {
	gescal(m, n, beta, C, incRowC, incColC);
	return;
    }
    //  Kernels that prefer row-major C: for column-major C compute
    //  C^T = B^T A^T (C^T is row major), as BLIS does
    if (UGemmRowPreference<T>::value && incRowC == IndexType(1) &&
        incColC != IndexType(1)) {
	gemm(n, m, k, alpha, conjB, B, incColB, incRowB, conjA, A, incColA,
	     incRowA, beta, C, incColC, incRowC);
	return;
    }
    //  The other kernels prefer column-major C (they update C directly only
    //  with unit row stride): for row-major C (e.g. the strided interface
    //  with a row-major C) also C^T = B^T A^T; about twice as fast
    if (!UGemmRowPreference<T>::value && incColC == IndexType(1) &&
        incRowC != IndexType(1)) {
	gemm(n, m, k, alpha, conjB, B, incColB, incRowB, conjA, A, incColA,
	     incRowA, beta, C, incColC, incRowC);
	return;
    }
    if (!conjA && !conjB &&
        gemm_direct(m, n, k, alpha, A, incRowA, incColA, B, incRowB, incColB,
                    beta, C, incRowC, incColC)) {
	return;
    }

    const GemmProblem<IndexType> problem = {m,       n,       k,       incRowA,
                                            incColA, incRowB, incColB, incRowC,
                                            incColC, conjA,   conjB,   1};
    const GemmPlan<IndexType, Alpha, TA, TB, Beta, TC> plan =
        gemmSelect<IndexType, Alpha, TA, TB, Beta, TC>(problem);

    plan.run(m, n, k, alpha, conjA, A, incRowA, incColA, conjB, B, incRowB,
             incColB, beta, C, incRowC, incColC);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_TCC
