#ifndef ULMBLAS_LAPACK_TRTRS_H
#define ULMBLAS_LAPACK_TRTRS_H 1

namespace ulmBLAS {

//
//  Solves op(A) X = B with the triangular n x n matrix A: op(A) = A, A^T
//  (trans) or A^H (trans and conj); conj without trans: conj(A).  unitDiag:
//  the diagonal of A is 1 and not referenced.  B is n x nrhs and
//  overwritten by X.  Returns 0, or i if A(i-1,i-1) is exactly zero (then
//  B is not changed, as LAPACK's ?trtrs).  Any strides.
//
template <typename IndexType, typename T>
    IndexType
    trtrs(bool         lower,
          bool         trans,
          bool         conj,
          bool         unitDiag,
          IndexType    n,
          IndexType    nrhs,
          const T      *A,
          IndexType    incRowA,
          IndexType    incColA,
          T            *B,
          IndexType    incRowB,
          IndexType    incColB);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_TRTRS_H

#include <ulmblas/lapack/trtrs.tcc>
