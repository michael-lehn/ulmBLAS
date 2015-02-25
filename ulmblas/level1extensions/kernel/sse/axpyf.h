#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_AXPYF_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_AXPYF_H

#include <ulmblas/config/fusefactor.h>
#include <ulmblas/level1extensions/kernel/ref/axpyf.h>
#include <type_traits>


namespace ulmBLAS { namespace sse {

using ref::axpyf;
using ref::acxpyf;

template <typename IndexType>
    typename std::enable_if<std::is_integral<IndexType>::value
                         && FuseFactor<double>::axpyf==2,
    void>::type
    axpyf(IndexType      n,
          const double   &alpha,
          const double   *a,
          IndexType      incA,
          const double   *X,
          IndexType      incRowX,
          IndexType      incColX,
          double         *y,
          IndexType      incY);

template <typename IndexType>
    typename std::enable_if<std::is_integral<IndexType>::value
                         && FuseFactor<double>::axpyf==4,
    void>::type
    axpyf(IndexType      n,
          const double   &alpha,
          const double   *a,
          IndexType      incA,
          const double   *X,
          IndexType      incRowX,
          IndexType      incColX,
          double         *y,
          IndexType      incY);

} } // namespace sse, ulmBLAS

#include <ulmblas/level1extensions/kernel/sse/axpyf.tcc>

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_AXPYF_H
