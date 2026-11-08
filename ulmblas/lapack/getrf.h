#ifndef ULMBLAS_LAPACK_GETRF_H
#define ULMBLAS_LAPACK_GETRF_H 1

namespace ulmBLAS {

//
//  LU factorization with partial pivoting A = P L U of an m x n matrix (L
//  unit lower, U upper triangular; both stored in A).  Any strides (A(i,j)
//  is A[i*incRowA + j*incColA]).  piv: min(m,n) entries with increment
//  incPiv, 0-based: row i was interchanged with row piv[i*incPiv].  Pivots
//  are chosen by |x| for real and by |re(x)| + |im(x)| for complex x (as
//  LAPACK).  Returns 0 or i if U(i-1,i-1) is exactly zero (1-based i); the
//  factorization is completed in any case.
//
//  Compiled with OpenMP, larger problems are factored by a task graph with
//  omp_get_max_threads() threads; inside an active parallel region and
//  without OpenMP by the serial recursive algorithm (getrf2).
//
template <typename IndexType, typename T, typename PivIndex>
    IndexType
    getrf(IndexType    m,
          IndexType    n,
          T            *A,
          IndexType    incRowA,
          IndexType    incColA,
          PivIndex     *piv,
          IndexType    incPiv);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRF_H

#include <ulmblas/lapack/getrf.tcc>
