#ifndef ULMBLAS_LEVEL1EXTENSIONS_GECOPY_H
#define ULMBLAS_LEVEL1EXTENSIONS_GECOPY_H 1

namespace ulmBLAS {

template <typename IndexType, typename MX, typename MY>
    void
    gecopy(IndexType      m,
           IndexType      n,
           bool           conjX,
           const MX       *X,
           IndexType      incRowX,
           IndexType      incColX,
           MY             *Y,
           IndexType      incRowY,
           IndexType      incColY);

template <typename IndexType, typename MX, typename MY>
    void
    gecopy(IndexType      m,
           IndexType      n,
           const MX       *X,
           IndexType      incRowX,
           IndexType      incColX,
           MY             *Y,
           IndexType      incRowY,
           IndexType      incColY);


} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_GECOPY_H 1

#include <ulmblas/level1extensions/gecopy.tcc>
