#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_AXPYF_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_AXPYF_H 1

#include <ulmblas/config/fusefactor.h>
#include <type_traits>

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename Alpha, typename VA, typename VX,
          typename VY>
    typename std::enable_if<std::is_integral<IndexType>::value
                  && FuseFactor<decltype(Alpha(0)*VA(0)*VX(0)+VY(0))>::axpyf==4,
    void>::type
    axpyf(IndexType      n,
          const Alpha    &alpha,
          const VA       *a,
          IndexType      incA,
          const VX       *x,
          IndexType      incRowX,
          IndexType      incColX,
          VY             *y,
          IndexType      incY);

template <typename IndexType, typename Alpha, typename VA, typename VX,
          typename VY>
    typename std::enable_if<std::is_integral<IndexType>::value
                  && FuseFactor<decltype(Alpha(0)*VA(0)*VX(0)+VY(0))>::axpyf!=4,
    void>::type
    axpyf(IndexType      n,
          const Alpha    &alpha,
          const VA       *a,
          IndexType      incA,
          const VX       *x,
          IndexType      incRowX,
          IndexType      incColX,
          VY             *y,
          IndexType      incY);

template <typename IndexType, typename Alpha, typename VA, typename VX,
          typename VY>
    typename std::enable_if<std::is_integral<IndexType>::value
                 && FuseFactor<decltype(Alpha(0)*VA(0)*VX(0)+VY(0))>::acxpyf==4,
    void>::type
    acxpyf(IndexType      n,
           const Alpha    &alpha,
           const VA       *a,
           IndexType      incA,
           const VX       *x,
           IndexType      incRowX,
           IndexType      incColX,
           VY             *y,
           IndexType      incY);

template <typename IndexType, typename Alpha, typename VA, typename VX,
          typename VY>
    typename std::enable_if<std::is_integral<IndexType>::value
                 && FuseFactor<decltype(Alpha(0)*VA(0)*VX(0)+VY(0))>::acxpyf!=4,
    void>::type
    acxpyf(IndexType      n,
           const Alpha    &alpha,
           const VA       *a,
           IndexType      incA,
           const VX       *x,
           IndexType      incRowX,
           IndexType      incColX,
           VY             *y,
           IndexType      incY);


} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_AXPYF_H

#include <ulmblas/level1extensions/kernel/ref/axpyf.tcc>
