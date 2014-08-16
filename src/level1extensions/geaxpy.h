#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_GEAXPY_H
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_GEAXPY_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename MX, typename MY>
    void
    geaxpy(IndexType      m,
           IndexType      n,
           const Alpha    &alpha,
           const MX       *X,
           IndexType      incRowX,
           IndexType      incColX,
           MY             *Y);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_GEAXPY_H 1

#include <src/level1extensions/geaxpy.tcc>
