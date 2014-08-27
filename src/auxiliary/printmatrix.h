#ifndef SRC_AUXILIARY_PRINTMATRIX_H
#define SRC_AUXILIARY_PRINTMATRIX_H 1

namespace ulmBLAS {

template <typename T, typename IndexType>
    void
    printMatrix(IndexType m, IndexType n,
                const T *X, IndexType incRowX, IndexType incColX);

} // namespace ulmBLAS

#endif // SRC_AUXILIARY_PRINTMATRIX_H

#include <src/auxiliary/printmatrix.tcc>
