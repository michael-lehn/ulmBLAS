#ifndef ULMBLAS_LAPACK_GETRF2_H
#define ULMBLAS_LAPACK_GETRF2_H 1

namespace ulmBLAS {

//
//  Recursive LU factorization with partial pivoting A = P L U of an m x n
//  matrix (as LAPACK's ?getrf2, with getf2 as base case), serial.  Any
//  strides.  piv: min(m,n) entries with increment incPiv, 0-based: row i
//  was interchanged with row piv[i*incPiv].  Pivots are chosen by |x| for
//  real and by |re(x)| + |im(x)| for complex x (as LAPACK's i?amax).
//  Returns 0 or i if U(i-1,i-1) is exactly zero (1-based i); the
//  factorization is completed in any case.
//
template <typename IndexType, typename T, typename PivIndex>
IndexType
getrf2(IndexType m, IndexType n, T *A, IndexType incRowA, IndexType incColA,
       PivIndex *piv, IndexType incPiv);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRF2_H

#include <ulmblas/lapack/getrf2.tcc>
