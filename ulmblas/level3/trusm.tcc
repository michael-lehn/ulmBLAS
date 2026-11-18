#ifndef ULMBLAS_LEVEL3_TRUSM_TCC
#define ULMBLAS_LEVEL3_TRUSM_TCC 1

#include <cstddef>
#include <ulmblas/level3/fpsrscope.h>
#include <ulmblas/level3/gemm/frametrsm.h>
#include <ulmblas/level3/gemm/select.h>
#include <ulmblas/level3/trusm.h>

namespace ulmBLAS {

//
//  B <- alpha A^{-1} B for the upper triangular m x m matrix A: the
//  frame algorithm of trsm (ulmblas/level3/gemm/frametrsm.h) with the frame
//  that the selection table of gemm has for an m x n x m product
//  (gemm/select.h; level3Select with TrsmOp).
//
template <typename IndexType, typename Alpha, typename TA, typename TB>
void
trusm(IndexType m, IndexType n, const Alpha &alpha, bool conjA, bool unitDiag,
      const TA *A, IndexType incRowA, IndexType incColA, TB *B,
      IndexType incRowB, IndexType incColB)
{
    const FpsrScope fpsrScope; // see level3/fpsrscope.h

    typedef decltype(Alpha(0) * TA(0) * TB(0)) T;

    if (m <= 0 || n <= 0) {
	return;
    }
    const TrsmPlan<IndexType, Alpha, TA, TB> plan =
        level3Select<T, TrsmOp<IndexType, Alpha, TA, TB>>(
            std::ptrdiff_t(m), std::ptrdiff_t(n), std::ptrdiff_t(m));
    plan.run(false, m, n, alpha, conjA, unitDiag, A, incRowA, incColA, B,
             incRowB, incColB);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_TRUSM_TCC
