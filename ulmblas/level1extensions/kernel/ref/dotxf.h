#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTXF_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTXF_H 1

#include <ulmblas/config/fusefactor.h>
#include <type_traits>

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename TX, typename TY, typename Result>
    typename std::enable_if<std::is_integral<IndexType>::value
                      && FuseFactor<decltype(TX(0)*TY(0)+Result(0))>::dotuxf==4,
    void>::type
    dotuxf(IndexType      n,
           const TX       *X,
           IndexType      incRowX,
           IndexType      incColX,
           const TY       *y,
           IndexType      incY,
           Result         *result,
           IndexType      resultInc);

template <typename IndexType, typename TX, typename TY, typename Result>
    typename std::enable_if<std::is_integral<IndexType>::value
                      && FuseFactor<decltype(TX(0)*TY(0)+Result(0))>::dotuxf!=4,
    void>::type
    dotuxf(IndexType      n,
           const TX       *X,
           IndexType      incRowX,
           IndexType      incColX,
           const TY       *y,
           IndexType      incY,
           Result         *result,
           IndexType      resultInc);

template <typename IndexType, typename TX, typename TY, typename Result>
    typename std::enable_if<std::is_integral<IndexType>::value
                      && FuseFactor<decltype(TX(0)*TY(0)+Result(0))>::dotcxf==4,
    void>::type
    dotcxf(IndexType      n,
           const TX       *X,
           IndexType      incRowX,
           IndexType      incColX,
           const TY       *y,
           IndexType      incY,
           Result         *result,
           IndexType      resultInc);

template <typename IndexType, typename TX, typename TY, typename Result>
    typename std::enable_if<std::is_integral<IndexType>::value
                      && FuseFactor<decltype(TX(0)*TY(0)+Result(0))>::dotcxf!=4,
    void>::type
    dotcxf(IndexType      n,
           const TX       *X,
           IndexType      incRowX,
           IndexType      incColX,
           const TY       *y,
           IndexType      incY,
           Result         *result,
           IndexType      resultInc);

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTXF_H

#include <ulmblas/level1extensions/kernel/ref/dotxf.tcc>
