#ifndef ULMBLAS_LEVEL3_HEUR2K_TCC
#define ULMBLAS_LEVEL3_HEUR2K_TCC 1

#include <complex>
#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1extensions/truscal.h>
#include <ulmblas/level3/heur2k.h>
#include <ulmblas/level3/fpsrscope.h>
#include <ulmblas/level3/gemmtr.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
heur2k(IndexType n, IndexType k, const Alpha &alpha, const TA *A,
       IndexType incRowA, IndexType incColA, const TB *B, IndexType incRowB,
       IndexType incColB, const Beta &beta, TC *C, IndexType incRowC,
       IndexType incColC)
{
    const FpsrScope fpsrScope; // see level3/fpsrscope.h

    if (n == 0 || ((alpha == Alpha(0) || k == 0) && beta == Beta(1))) {
	return;
    }

    if (alpha == Alpha(0) || k == 0) {
	truscal(n, n, false, beta, C, incRowC, incColC);
	for (IndexType i = 0; i < n; ++i) {
	    C[i * (incRowC + incColC)] = std::real(C[i * (incRowC + incColC)]);
	}
	return;
    }

    //  C <- beta C + alpha A B^H + conj(alpha) B A^H in the upper triangle:
    //  two gemmt calls (ulmblas/level3/gemmtr.tcc, the frame algorithm of
    //  gemmt).  beta stays real: a complex beta would turn 0*NaN in the
    //  imaginary part into NaN in the real part.
    gemmtr(false, n, k, alpha, false, A, incRowA, incColA, true, B, incColB,
           incRowB, beta, C, incRowC, incColC);
    gemmtr(false, n, k, conjugate(alpha), false, B, incRowB, incColB, true, A,
           incColA, incRowA, Beta(1), C, incRowC, incColC);
    //  the diagonal of a Hermitian matrix is real
    for (IndexType i = 0; i < n; ++i) {
	C[i * (incRowC + incColC)] = std::real(C[i * (incRowC + incColC)]);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_HEUR2K_TCC
