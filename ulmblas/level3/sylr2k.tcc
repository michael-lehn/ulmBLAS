#ifndef ULMBLAS_LEVEL3_SYLR2K_TCC
#define ULMBLAS_LEVEL3_SYLR2K_TCC 1

#include <ulmblas/level1extensions/trlscal.h>
#include <ulmblas/level3/sylr2k.h>
#include <ulmblas/level3/fpsrscope.h>
#include <ulmblas/level3/gemmtr.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
sylr2k(IndexType n, IndexType k, const Alpha &alpha, const TA *A,
       IndexType incRowA, IndexType incColA, const TB *B, IndexType incRowB,
       IndexType incColB, const Beta &beta, TC *C, IndexType incRowC,
       IndexType incColC)
{
    const FpsrScope fpsrScope; // see level3/fpsrscope.h

    if (n == 0 || ((alpha == Alpha(0) || k == 0) && beta == Beta(1))) {
	return;
    }

    if (alpha == Alpha(0) || k == 0) {
	trlscal(n, n, false, beta, C, incRowC, incColC);
	return;
    }

    //  C <- beta C + alpha A B^T + alpha B A^T in the lower triangle: two
    //  gemmt calls (ulmblas/level3/gemmtr.tcc, the frame algorithm of gemmt)
    gemmtr(true, n, k, alpha, false, A, incRowA, incColA, false, B, incColB,
           incRowB, beta, C, incRowC, incColC);
    gemmtr(true, n, k, alpha, false, B, incRowB, incColB, false, A, incColA,
           incRowA, Beta(1), C, incRowC, incColC);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_SYLR2K_TCC
