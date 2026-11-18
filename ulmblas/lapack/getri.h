#ifndef ULMBLAS_LAPACK_GETRI_H
#define ULMBLAS_LAPACK_GETRI_H 1

namespace ulmBLAS {

//
//  Inverse of an n x n matrix from its LU factorization P A = L U (A and
//  piv as returned by getrf, with increment incPiv; piv 0-based, or
//  pivBase-based, e.g. 1 for the pivots of LAPACK): inv(A) = inv(U) inv(L)
//  P, in place.  Returns 0, or i if U(i-1,i-1) is exactly zero (then A is
//  not changed).  No work space: inv(U) and inv(L) in place (trtri), then
//  their product in place.  With OpenMP the inverses of larger matrices as
//  tasks.  Any strides.
//
template <typename IndexType, typename T, typename PivIndex>
IndexType
getri(IndexType n, T *A, IndexType incRowA, IndexType incColA,
      const PivIndex *piv, IndexType incPiv, IndexType pivBase = IndexType(0));

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRI_H

#include <ulmblas/lapack/getri.tcc>
