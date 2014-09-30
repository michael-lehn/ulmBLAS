#ifndef ULMBLAS_LEVEL3_UGEMM_H
#define ULMBLAS_LEVEL3_UGEMM_H 1

namespace ulmBLAS {

template <typename T>
    int
    ugemm_mr();

template <typename T>
    int
    ugemm_nr();

//
//  Buffered variant.  Used for zero padded panels.
//
template <typename IndexType, typename T, typename Beta, typename TC>
    void
    ugemm(IndexType    mr,
          IndexType    nr,
          IndexType    kc,
          const T      &alpha,
          const T      *A,
          const T      *B,
          const Beta   &beta,
          TC           *C,
          IndexType    incRowC,
          IndexType    incColC,
          const T      *nextA,
          const T      *nextB);

//
//  Buffered variant.  Used if the result alpha*A*B needs to be upcasted for
//  computing C <- beta*C + (alpha*A*B)
//
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

//
//  Unbuffered variant.
//
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

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UGEMM_H

#include <ulmblas/level3/ugemm.tcc>
