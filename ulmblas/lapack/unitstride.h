#ifndef ULMBLAS_LAPACK_UNITSTRIDE_H
#define ULMBLAS_LAPACK_UNITSTRIDE_H 1

#include <cstddef>

namespace ulmBLAS {

//
//  The largest copy of a block of a matrix in the LAPACK functions, in
//  elements: KC x NC of the gemm block sizes (config/blocksize.h), or
//  ULMBLAS_LAPACK_COPY_MAX if defined.  No copy (and no other work space)
//  of the LAPACK functions grows with the matrix dimensions.
//
template <typename T>
constexpr std::size_t
lapackCopyMax();

//  true if an m x n block can be copied (m, n >= 0)
template <typename T, typename IndexType>
bool
lapackCopyFits(IndexType m, IndexType n);

//
//  Calls f(B, incRowB, incColB) on the m x n matrix A.  The unblocked
//  factorizations are fast if one of the strides is 1 (column major, row
//  major, or a transposed view).  If neither is and A fits into a copy
//  (lapackCopyFits), f works on a contiguous column-major copy of A, which
//  is copied back afterwards: O(m n) copies for O(m n min(m,n))
//  operations, like the BLAS kernels pack their operands.  Otherwise f
//  works on A itself.  lowerOnly: only the lower triangle is copied (and
//  copied back); the other one is not referenced.  Returns the result of
//  f.
//
template <typename IndexType, typename T, typename F>
IndexType
withUnitStride(IndexType m, IndexType n, T *A, IndexType incRowA,
               IndexType incColA, bool lowerOnly, F f);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_UNITSTRIDE_H

#include <ulmblas/lapack/unitstride.tcc>
