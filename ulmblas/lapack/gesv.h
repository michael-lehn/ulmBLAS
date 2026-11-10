#ifndef ULMBLAS_LAPACK_GESV_H
#define ULMBLAS_LAPACK_GESV_H 1

namespace ulmBLAS {

//
//  Solves A X = B: LU factorization of A (getrf), then getrs.  A is
//  overwritten by L and U, piv by the pivots (0-based, increment incPiv), B
//  (n x nrhs) by X.  Returns the INFO of getrf: if U(i-1,i-1) is exactly
//  zero (i > 0), A is singular and B is not changed.  Any strides.
//
template <typename IndexType, typename T, typename PivIndex>
IndexType
gesv(IndexType n, IndexType nrhs, T *A, IndexType incRowA, IndexType incColA,
     PivIndex *piv, IndexType incPiv, T *B, IndexType incRowB,
     IndexType incColB);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GESV_H

#include <ulmblas/lapack/gesv.tcc>
