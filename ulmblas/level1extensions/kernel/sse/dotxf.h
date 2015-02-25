#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXF_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXF_H 1

#include <ulmblas/config/fusefactor.h>
#include <ulmblas/level1extensions/kernel/ref/dotxf.h>
#include <type_traits>

namespace ulmBLAS { namespace sse {

using ref::dotuxf;
using ref::dotcxf;

template <typename IndexType>
    typename std::enable_if<std::is_integral<IndexType>::value
                         && FuseFactor<double>::dotuxf==2,
    void>::type
    dotuxf(IndexType      n,
           const double   *X,
           IndexType      incRowX,
           IndexType      incColX,
           const double   *y,
           IndexType      incY,
           double         *result,
           IndexType      resultInc);

template <typename IndexType>
    typename std::enable_if<std::is_integral<IndexType>::value
                         && FuseFactor<double>::dotuxf==4,
    void>::type
    dotuxf(IndexType      n,
           const double   *X,
           IndexType      incRowX,
           IndexType      incColX,
           const double   *y,
           IndexType      incY,
           double         *result,
           IndexType      resultInc);

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXF_H

#include <ulmblas/level1extensions/kernel/sse/dotxf.tcc>
