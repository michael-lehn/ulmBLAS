#ifndef ULMBLAS_LEVEL3_GEMMTR_TCC
#define ULMBLAS_LEVEL3_GEMMTR_TCC 1

#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/level3/fpsrscope.h>
#include <ulmblas/level1extensions/trlaxpy.h>
#include <ulmblas/level1extensions/trlscal.h>
#include <ulmblas/level1extensions/truaxpy.h>
#include <ulmblas/level1extensions/truscal.h>
#include <ulmblas/level3/gemm.h>
#include <ulmblas/level3/gemmtr.h>

namespace ulmBLAS {

//
//  C is processed in block columns of NB columns.  For each block column:
//
//  - the part of the triangle outside the diagonal block (below it for the
//    lower, above it for the upper triangle) is an ordinary product and is
//    computed by gemm directly in C;
//  - the diagonal block D = alpha*A(rows of the block,:)*B(:,columns of
//    the block) is computed by gemm in a buffer (beta = 0: the buffer is
//    not read), then the triangle of the block of C is scaled by beta and
//    the triangle of D is added.
//
//  So the products use the kernels of gemm, and C outside the triangle is
//  not accessed.  The diagonal blocks cost NB*NB*k multiplications more
//  than necessary (half of each diagonal block), about NB/n of the total.
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

    constexpr IndexType NB = 64;

    static MemoryPool<T> memoryPool;

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

    T *D = memoryPool.allocate(NB * NB);

    for (IndexType j = 0; j < n; j += NB) {
	const IndexType nb = (n - j < NB) ? n - j : NB;

	//  the block column outside the diagonal block
	if (lower && j + nb < n) {
	    gemm(n - j - nb, nb, k, alpha, conjA, &A[(j + nb) * incRowA],
	         incRowA, incColA, conjB, &B[j * incColB], incRowB, incColB,
	         beta, &C[(j + nb) * incRowC + j * incColC], incRowC, incColC);
	} else if (!lower && j > 0) {
	    gemm(j, nb, k, alpha, conjA, A, incRowA, incColA, conjB,
	         &B[j * incColB], incRowB, incColB, beta, &C[j * incColC],
	         incRowC, incColC);
	}

	//  the diagonal block: D = alpha*A(j:j+nb,:)*B(:,j:j+nb), then
	//  C(triangle) = beta*C(triangle) + D(triangle)
	TC *Cjj = &C[j * incRowC + j * incColC];
	gemm(nb, nb, k, alpha, conjA, &A[j * incRowA], incRowA, incColA, conjB,
	     &B[j * incColB], incRowB, incColB, T(0), D, IndexType(1), nb);
	if (lower) {
	    trlscal(nb, nb, false, beta, Cjj, incRowC, incColC);
	    trlaxpy(nb, nb, false, T(1), D, IndexType(1), nb, Cjj, incRowC,
	            incColC);
	} else {
	    truscal(nb, nb, false, beta, Cjj, incRowC, incColC);
	    truaxpy(nb, nb, false, T(1), D, IndexType(1), nb, Cjj, incRowC,
	            incColC);
	}
    }

    memoryPool.release(D);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMMTR_TCC
