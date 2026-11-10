#ifndef ULMBLAS_LAPACK_LAUUM_H
#define ULMBLAS_LAPACK_LAUUM_H 1

namespace ulmBLAS {

//
//  Product of a triangular n x n matrix with its conjugate transpose, in
//  place: upper: A := U U^H, lower: A := L^H L (only that triangle is
//  referenced and written; the diagonal of U or L is real, as in LAPACK's
//  ?lauum).  Recursive; with OpenMP larger matrices as tasks.  Any strides.
//
template <typename IndexType, typename T>
void
lauum(bool lower, IndexType n, T *A, IndexType incRowA, IndexType incColA);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_LAUUM_H

#include <ulmblas/lapack/lauum.tcc>
