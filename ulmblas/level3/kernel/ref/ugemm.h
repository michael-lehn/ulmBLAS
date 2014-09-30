#ifndef ULMBLAS_LEVEL3_KERNEL_REF_UGEMM_H
#define ULMBLAS_LEVEL3_KERNEL_REF_UGEMM_H 1

namespace ulmBLAS { namespace ref {

template <typename T>
    int
    ugemm_mr();

template <typename T>
    int
    ugemm_nr();

template <typename IndexType, typename T, typename Beta, typename TC>
    void
    ugemm(IndexType   kc,
          const T     &alpha,
          const T     *A,
          const T     *B,
          const Beta  &beta,
          TC          *C,
          IndexType   incRowC,
          IndexType   incColC,
          const T     *nextA,
          const T     *nextB);


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

#endif // ULMBLAS_LEVEL3_KERNEL_REF_AXPY_H 1

#include <ulmblas/level3/kernel/ref/ugemm.tcc>
