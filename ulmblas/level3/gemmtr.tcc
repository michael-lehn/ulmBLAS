#ifndef ULMBLAS_LEVEL3_GEMMTR_TCC
#define ULMBLAS_LEVEL3_GEMMTR_TCC 1

#include <ulmblas/level3/fpsrscope.h>
#include <ulmblas/level1extensions/trlscal.h>
#include <ulmblas/level1extensions/truscal.h>
#include <ulmblas/level3/gemm/default.h>
#include <ulmblas/level3/gemm/frametr.h>
#include <ulmblas/level3/gemm/select.h>
#include <ulmblas/level3/ukernel/ugemm.h>
#include <ulmblas/level3/gemmtr.h>

namespace ulmBLAS {

//
//  C <- beta C + alpha A B for the lower (upper) triangle of the n x n
//  matrix C: the special cases here, then the frame algorithm of gemmt
//  (ulmblas/level3/gemm/frametr.h) with the frame that the selection table
//  of gemm has for an n x n x k product (gemm/select.h; level3Select with
//  GemmtOp); C outside the triangle is not accessed.
//
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
gemmtr(bool lower, IndexType n, IndexType k, const Alpha &alpha, bool conjA,
       const TA *A, IndexType incRowA, IndexType incColA, bool conjB,
       const TB *B, IndexType incRowB, IndexType incColB, const Beta &beta,
       TC *C, IndexType incRowC, IndexType incColC)
{
    const FpsrScope fpsrScope; // see level3/fpsrscope.h

    typedef decltype(Alpha(0) * TA(0) * TB(0)) T;

    if (n <= 0) {
	return;
    }
    if (alpha == Alpha(0) || k <= 0) {
	if (lower) {
	    trlscal(n, n, false, beta, C, incRowC, incColC);
	} else {
	    truscal(n, n, false, beta, C, incRowC, incColC);
	}
	return;
    }
    //  as gemm (ulmblas/level3/gemm.tcc): C^T = B^T A^T, the other
    //  triangle, if the kernels prefer row-major C and C is column major,
    //  or the other way round
    const bool rowPref = UGemmRowPreference<T>::value;
    if ((rowPref && incRowC == IndexType(1) && incColC != IndexType(1)) ||
        (!rowPref && incColC == IndexType(1) && incRowC != IndexType(1))) {
	gemmtr(!lower, n, k, alpha, conjB, B, incColB, incRowB, conjA, A,
	       incColA, incRowA, beta, C, incColC, incRowC);
	return;
    }
    const GemmtPlan<IndexType, Alpha, TA, TB, Beta, TC> plan =
        level3Select<T, GemmtOp<IndexType, Alpha, TA, TB, Beta, TC>>(
            std::ptrdiff_t(n), std::ptrdiff_t(n), std::ptrdiff_t(k));
    plan.run(lower, n, k, alpha, conjA, A, incRowA, incColA, conjB, B, incRowB,
             incColB, beta, C, incRowC, incColC);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMMTR_TCC
