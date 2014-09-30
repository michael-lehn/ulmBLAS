#ifndef ULMBLAS_LEVEL1EXTENSIONS_TRUAXPY_H
#define ULMBLAS_LEVEL1EXTENSIONS_TRUAXPY_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename MX, typename MY>
    void
    truaxpy(IndexType    m,
            IndexType    n,
            bool         unit,
            const Alpha  &alpha,
            MX           *X,
            IndexType    incRowX,
            IndexType    incColX,
            MY           *Y,
            IndexType    incRowY,
            IndexType    incColY);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_TRUAXPY_H 1

#include <ulmblas/level1extensions/truaxpy.tcc>
