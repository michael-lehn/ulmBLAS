#ifndef ULMBLAS_LEVEL3_SYLRK_TCC
#define ULMBLAS_LEVEL3_SYLRK_TCC 1

#include <ulmblas/level1extensions/trlscal.h>
#include <ulmblas/level3/sylrk.h>
#include <ulmblas/level3/fpsrscope.h>
#include <ulmblas/level3/gemmtr.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename Beta,
          typename TC>
void
sylrk(IndexType n, IndexType k, const Alpha &alpha, const TA *A,
      IndexType incRowA, IndexType incColA, const Beta &beta, TC *C,
      IndexType incRowC, IndexType incColC)
{
    const FpsrScope fpsrScope; // see level3/fpsrscope.h

    if (n == 0 || ((alpha == Alpha(0) || k == 0) && beta == Beta(1))) {
	return;
    }

    if (alpha == Alpha(0) || k == 0) {
	trlscal(n, n, false, beta, C, incRowC, incColC);
	return;
    }

    //  C <- beta C + alpha A A^T in the lower triangle: gemmt with B = A^T
    //  (ulmblas/level3/gemmtr.tcc, the frame algorithm of gemmt)
    gemmtr(true, n, k, alpha, false, A, incRowA, incColA, false, A, incColA,
           incRowA, beta, C, incRowC, incColC);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_SYLRK_TCC
