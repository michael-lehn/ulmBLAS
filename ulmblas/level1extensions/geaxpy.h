#ifndef ULMBLAS_LEVEL1EXTENSIONS_GEAXPY_H
#define ULMBLAS_LEVEL1EXTENSIONS_GEAXPY_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename MX, typename MY>
    void
    geaxpy(IndexType      m,
           IndexType      n,
           const Alpha    &alpha,
           const MX       *X,
           IndexType      incRowX,
           IndexType      incColX,
           MY             *Y,
           IndexType      incRowY,
           IndexType      incColY);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_GEAXPY_H 1

#include <ulmblas/level1extensions/geaxpy.tcc>
