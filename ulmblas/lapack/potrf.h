#ifndef ULMBLAS_LAPACK_POTRF_H
#define ULMBLAS_LAPACK_POTRF_H 1

namespace ulmBLAS {

//
//  Cholesky factorization of a Hermitian (real: symmetric) positive
//  definite n x n matrix:
//
//      lower:  A = L L^H, L is stored in the lower triangle of A
//      upper:  A = U^H U, U is stored in the upper triangle of A
//
//  Only the given triangle is referenced.  Any strides (A(i,j) is
//  A[i*incRowA + j*incColA]).  Returns 0 or i if the leading minor of order
//  i is not positive definite (the factorization could not be completed).
//
//  Compiled with OpenMP, larger problems are factored by a task graph
//  (tiled, right-looking, as PLASMA) with numThreads() threads (threads.h);
//  inside an active parallel region and without OpenMP by the serial
//  recursive algorithm.
//
template <typename IndexType, typename T>
    IndexType
    potrf(bool         lower,
          IndexType    n,
          T            *A,
          IndexType    incRowA,
          IndexType    incColA);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_POTRF_H

#include <ulmblas/lapack/potrf.tcc>
