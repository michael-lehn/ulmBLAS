#ifndef ULMBLAS_LEVEL3_UKERNEL_REF_UGEMM_H
#define ULMBLAS_LEVEL3_UKERNEL_REF_UGEMM_H 1

#include <ulmblas/config/blocksize.h>

namespace ulmBLAS { namespace ref {

template <typename T>
struct BlockSizeUGemm
{
    static const int MR = BlockSize<T>::MR;
    static const int NR = BlockSize<T>::NR;

    static_assert(MR>0 && NR>0, "Invalid block size.");
};

template <typename IndexType, typename T>
    void
    ugemm(IndexType   kc,
          const T     &alpha,
          const T     *A,
          const T     *B,
          const T     &beta,
          T           *C,
          IndexType   incRowC,
          IndexType   incColC,
          const T     *nextA,
          const T     *nextB);

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_REF_UGEMM_H

#include <ulmblas/level3/ukernel/ref/ugemm.tcc>
