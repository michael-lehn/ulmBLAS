#ifndef ULMBLAS_LAPACK_POSV_H
#define ULMBLAS_LAPACK_POSV_H 1

namespace ulmBLAS {

//
//  Solves A X = B with the Hermitian (symmetric) positive definite matrix
//  A: Cholesky factorization (potrf, triangle lower or upper), then potrs.
//  Returns the INFO of potrf: if i > 0, the leading minor of order i is not
//  positive definite and B is not changed.  Any strides.
//
template <typename IndexType, typename T>
IndexType
posv(bool lower, IndexType n, IndexType nrhs, T *A, IndexType incRowA,
     IndexType incColA, T *B, IndexType incRowB, IndexType incColB);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_POSV_H

#include <ulmblas/lapack/posv.tcc>
