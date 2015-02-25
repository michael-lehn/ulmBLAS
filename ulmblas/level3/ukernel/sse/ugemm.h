#ifndef ULMBLAS_LEVEL3_UKERNEL_SSE_UGEMM_H
#define ULMBLAS_LEVEL3_UKERNEL_SSE_UGEMM_H 1

#include <type_traits>
#include <ulmblas/level3/ukernel/ref/ugemm.h>

namespace ulmBLAS { namespace sse {

template <typename T>
struct BlockSizeUGemm
{
    static const int MR = (std::is_same<T,double>::value) ? 4
                        : ref::BlockSizeUGemm<T>::MR;
    static const int NR = (std::is_same<T,double>::value) ? 4
                        : ref::BlockSizeUGemm<T>::NR;

    static_assert(MR>0 && NR>0, "Invalid block size.");
};

using ref::ugemm;

template <typename IndexType>
static typename std::enable_if<std::is_convertible<IndexType,long>::value,
    void>::type
    ugemm(IndexType      kc_,
          const double   &alpha,
          const double   *A,
          const double   *B,
          const double   &beta,
          double         *C,
          IndexType      incRowC_,
          IndexType      incColC_,
          const double   *nextA,
          const double   *nextB);

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SSE_UGEMM_H

#include <ulmblas/level3/ukernel/sse/ugemm.tcc>
