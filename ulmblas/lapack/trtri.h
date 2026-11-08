#ifndef ULMBLAS_LAPACK_TRTRI_H
#define ULMBLAS_LAPACK_TRTRI_H 1

namespace ulmBLAS {

//
//  Inverse of a triangular n x n matrix, in place (lower or upper
//  triangle; unitDiag: the diagonal is 1 and not referenced).  Returns 0,
//  or i if A(i-1,i-1) is exactly zero (then A is not changed, as LAPACK's
//  ?trtri).  Recursive, serial.  Any strides.
//
template <typename IndexType, typename T>
    IndexType
    trtri(bool         lower,
          bool         unitDiag,
          IndexType    n,
          T            *A,
          IndexType    incRowA,
          IndexType    incColA);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_TRTRI_H

#include <ulmblas/lapack/trtri.tcc>
