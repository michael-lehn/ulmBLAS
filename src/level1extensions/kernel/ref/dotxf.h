#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_REF_DOTXF_H
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_REF_DOTXF_H 1

namespace ulmBLAS { namespace ref {

template <typename T>
    int
    dotuxf_fusefactor();

template <typename IndexType, typename TX, typename TY, typename Result>
    void
    dotuxf(IndexType      n,
           const TX       *X,
           IndexType      incRowX,
           IndexType      incColX,
           const TY       *y,
           IndexType      incY,
           Result         *result,
           IndexType      resultInc);

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_KERNEL_REF_DOTXF_H

#include <src/level1extensions/kernel/ref/dotxf.tcc>
