#ifndef ULMBLAS_AUXILIARY_PRINTMATRIX_H
#define ULMBLAS_AUXILIARY_PRINTMATRIX_H 1

namespace ulmBLAS {

template <typename T, typename IndexType>
    void
    printMatrix(IndexType m, IndexType n,
                const T *X, IndexType incRowX, IndexType incColX);

template <typename T, typename IndexType>
    void
    printSylMatrix(IndexType m,
                   const T *X, IndexType incRowX, IndexType incColX);

template <typename T, typename IndexType>
    void
    printSyuMatrix(IndexType m,
                   const T *X, IndexType incRowX, IndexType incColX);

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_PRINTMATRIX_H

#include <ulmblas/auxiliary/printmatrix.tcc>
