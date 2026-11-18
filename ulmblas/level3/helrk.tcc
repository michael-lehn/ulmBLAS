#ifndef ULMBLAS_LEVEL3_HELRK_TCC
#define ULMBLAS_LEVEL3_HELRK_TCC 1

#include <complex>
#include <ulmblas/level1extensions/trlscal.h>
#include <ulmblas/level3/helrk.h>
#include <ulmblas/level3/fpsrscope.h>
#include <ulmblas/level3/gemmtr.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename Beta,
          typename TC>
void
helrk(IndexType n, IndexType k, const Alpha &alpha, const TA *A,
      IndexType incRowA, IndexType incColA, const Beta &beta, TC *C,
      IndexType incRowC, IndexType incColC)
{
    const FpsrScope fpsrScope; // see level3/fpsrscope.h

    typedef decltype(Alpha(0) * TA(0)) T;

    if (n == 0 || ((alpha == Alpha(0) || k == 0) && beta == Beta(1))) {
	return;
    }

    if (alpha == Alpha(0) || k == 0) {
	trlscal(n, n, false, beta, C, incRowC, incColC);
	for (IndexType i = 0; i < n; ++i) {
	    C[i * (incRowC + incColC)] = std::real(C[i * (incRowC + incColC)]);
	}
	return;
    }

    //  C <- beta C + alpha A A^H in the lower triangle: gemmt with B = A^H
    //  (ulmblas/level3/gemmtr.tcc, the frame algorithm of gemmt)
    gemmtr(true, n, k, T(alpha), false, A, incRowA, incColA, true, A, incColA,
           incRowA, beta, C, incRowC, incColC);
    //  the diagonal of a Hermitian matrix is real
    for (IndexType i = 0; i < n; ++i) {
	C[i * (incRowC + incColC)] = std::real(C[i * (incRowC + incColC)]);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_HELRK_TCC
