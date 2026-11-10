#ifndef ULMBLAS_LAPACK_GETRI_H
#define ULMBLAS_LAPACK_GETRI_H 1

namespace ulmBLAS {

//
//  Inverse of an n x n matrix from its LU factorization P A = L U (A and
//  piv as returned by getrf, piv 0-based with increment incPiv): inv(A) =
//  inv(U) inv(L) P, in place.  Returns 0, or i if U(i-1,i-1) is exactly
//  zero (then A is not changed).  Needs n x n elements of work space.
//  With OpenMP larger matrices as tasks.  Any strides.
//
template <typename IndexType, typename T, typename PivIndex>
IndexType
getri(IndexType n, T *A, IndexType incRowA, IndexType incColA,
      const PivIndex *piv, IndexType incPiv);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRI_H

#include <ulmblas/lapack/getri.tcc>
