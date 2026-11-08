#ifndef ULMBLAS_LAPACK_POTRI_H
#define ULMBLAS_LAPACK_POTRI_H 1

namespace ulmBLAS {

//
//  Inverse of a Hermitian (symmetric) positive definite matrix from its
//  Cholesky factorization (potrf): upper: inv(A) = inv(U) inv(U)^H, lower:
//  inv(L)^H inv(L), in the same triangle.  Returns 0, or i if the factor
//  has an exactly zero diagonal element i (1-based).  With OpenMP larger matrices as tasks.
//  Any strides.
//
template <typename IndexType, typename T>
    IndexType
    potri(bool         lower,
          IndexType    n,
          T            *A,
          IndexType    incRowA,
          IndexType    incColA);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_POTRI_H

#include <ulmblas/lapack/potri.tcc>
