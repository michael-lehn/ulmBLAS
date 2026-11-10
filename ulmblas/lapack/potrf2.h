#ifndef ULMBLAS_LAPACK_POTRF2_H
#define ULMBLAS_LAPACK_POTRF2_H 1

namespace ulmBLAS {

//
//  Recursive Cholesky factorization A = L L^H of the lower triangle (as
//  LAPACK's ?potrf2, with potf2 as base case), serial:
//
//      [ A11     ]   [ L11     ] [ L11^H L21^H ]
//      [ A21 A22 ] = [ L21 L22 ] [       L22^H ]
//
//  L11 = chol(A11),  L21 = A21 L11^{-H},  L22 = chol(A22 - L21 L21^H).
//
//  Any strides.  Returns 0 or i if the leading minor of order i (1-based)
//  is not positive definite.
//
template <typename IndexType, typename T>
IndexType
potrf2(IndexType n, T *A, IndexType incRowA, IndexType incColA);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_POTRF2_H

#include <ulmblas/lapack/potrf2.tcc>
