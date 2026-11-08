#ifndef ULMBLAS_LAPACK_POTRS_H
#define ULMBLAS_LAPACK_POTRS_H 1

namespace ulmBLAS {

//
//  Solves A X = B with the Cholesky factorization of potrf: lower: A = L
//  L^H, upper: A = U^H U (as stored by potrf).  B is n x nrhs and
//  overwritten by X.  Any strides.  Larger problems are solved by tasks.
//
template <typename IndexType, typename T>
    void
    potrs(bool         lower,
          IndexType    n,
          IndexType    nrhs,
          const T      *A,
          IndexType    incRowA,
          IndexType    incColA,
          T            *B,
          IndexType    incRowB,
          IndexType    incColB);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_POTRS_H

#include <ulmblas/lapack/potrs.tcc>
