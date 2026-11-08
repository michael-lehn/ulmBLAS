#ifndef ULMBLAS_LAPACK_GETRS_H
#define ULMBLAS_LAPACK_GETRS_H 1

namespace ulmBLAS {

//
//  Solves A X = B, A^T X = B (trans) or A^H X = B (trans and conj) with the
//  LU factorization P A = L U of getrf (A and piv as returned by getrf,
//  piv 0-based with increment incPiv).  B is n x nrhs and overwritten by X.
//  Any strides.  Larger problems are solved by tasks (OpenMP, see
//  tasks/solvetasks.h).
//
template <typename IndexType, typename T, typename PivIndex>
    void
    getrs(bool             trans,
          bool             conj,
          IndexType        n,
          IndexType        nrhs,
          const T          *A,
          IndexType        incRowA,
          IndexType        incColA,
          const PivIndex   *piv,
          IndexType        incPiv,
          T                *B,
          IndexType        incRowB,
          IndexType        incColB);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRS_H

#include <ulmblas/lapack/getrs.tcc>
