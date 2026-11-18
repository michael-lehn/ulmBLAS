#ifndef ULMBLAS_LEVEL3_SYLMM_TCC
#define ULMBLAS_LEVEL3_SYLMM_TCC 1

#include <cstddef>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/fpsrscope.h>
#include <ulmblas/level3/gemm/framesymm.h>
#include <ulmblas/level3/gemm/select.h>
#include <ulmblas/level3/sylmm.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
sylmm(IndexType m, IndexType n, const Alpha &alpha, const TA *A,
      IndexType incRowA, IndexType incColA, const TB *B, IndexType incRowB,
      IndexType incColB, const Beta &beta, TC *C, IndexType incRowC,
      IndexType incColC)
{
    const FpsrScope fpsrScope; // see level3/fpsrscope.h

    typedef decltype(Alpha(0) * TA(0) * TB(0)) T;

    if (m == 0 || n == 0 || (alpha == Alpha(0) && (beta == Beta(1)))) {
	return;
    }
    if (alpha == Alpha(0)) {
	gescal(m, n, beta, C, incRowC, incColC);
	return;
    }
    //  C <- beta C + alpha A B, A symmetric (its lower triangle): the frame
    //  algorithm of symm (ulmblas/level3/gemm/framesymm.h) with the frame
    //  that the selection table of gemm has for an m x n x m product
    const SymmPlan<IndexType, Alpha, TA, TB, Beta, TC> plan =
        level3Select<T, SymmOp<false, IndexType, Alpha, TA, TB, Beta, TC>>(
            std::ptrdiff_t(m), std::ptrdiff_t(n), std::ptrdiff_t(m));
    plan.run(true, true, m, n, alpha, A, incRowA, incColA, B, incRowB, incColB,
             beta, C, incRowC, incColC);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_SYLMM_TCC
