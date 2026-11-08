#ifndef ULMBLAS_LAPACK_UNITSTRIDE_H
#define ULMBLAS_LAPACK_UNITSTRIDE_H 1

namespace ulmBLAS {

//
//  Calls f(B, incRowB, incColB) on the m x n matrix A.  The unblocked
//  factorizations are fast if one of the strides is 1 (column major, row
//  major, or a transposed view).  If neither is, f works on a contiguous
//  column-major copy of A, which is copied back afterwards: O(m n) copies
//  for O(m n min(m,n)) operations, like the BLAS kernels pack their
//  operands.  lowerOnly: only the lower triangle is copied (and copied
//  back); the other one is not referenced.  Returns the result of f.
//
template <typename IndexType, typename T, typename F>
    IndexType
    withUnitStride(IndexType    m,
                   IndexType    n,
                   T            *A,
                   IndexType    incRowA,
                   IndexType    incColA,
                   bool         lowerOnly,
                   F            f);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_UNITSTRIDE_H

#include <ulmblas/lapack/unitstride.tcc>
