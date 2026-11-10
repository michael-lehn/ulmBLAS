#ifndef ULMBLAS_LEVEL3_GEMMTR_H
#define ULMBLAS_LEVEL3_GEMMTR_H 1

namespace ulmBLAS {

//
//  The lower (lower = true) or upper triangle of the n x n matrix C,
//  including the diagonal, is set to beta*C + alpha*A*B with an n x k
//  matrix A and a k x n matrix B (op(A), op(B) of the BLAS: transposition
//  by the strides, conjugation by conjA, conjB).  The other triangle of C
//  is neither read nor written; for beta = 0, C is not read.
//
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
gemmtr(bool lower, IndexType n, IndexType k, const Alpha &alpha, bool conjA,
       const TA *A, IndexType incRowA, IndexType incColA, bool conjB,
       const TB *B, IndexType incRowB, IndexType incColB, const Beta &beta,
       TC *C, IndexType incRowC, IndexType incColC);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMMTR_H

#include <ulmblas/level3/gemmtr.tcc>
