#ifndef ULMBLAS_LAPACK_POTF2_H
#define ULMBLAS_LAPACK_POTF2_H 1

namespace ulmBLAS {

//
//  Unblocked Cholesky factorization A = L L^H of the lower triangle of an
//  n x n matrix (L^H = L^T for real T).  Only the lower triangle is
//  referenced.  Returns 0 or j+1 if the j-th pivot (0-based) is not
//  positive or NaN (the factorization stops there, as LAPACK's ?potf2).
//
template <typename IndexType, typename T>
IndexType
potf2(IndexType n, T *A, IndexType incRowA, IndexType incColA);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_POTF2_H

#include <ulmblas/lapack/potf2.tcc>
