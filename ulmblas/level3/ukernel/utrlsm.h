#ifndef ULMBLAS_LEVEL3_UKERNEL_UTRLSM_H
#define ULMBLAS_LEVEL3_UKERNEL_UTRLSM_H 1

namespace ulmBLAS {

//
//  Buffered variant.  Used for zero padded panels.
//
template <typename IndexType, typename T, typename TC>
    void
    utrlsm(IndexType    mr,
           IndexType    nr,
           const T      *A,
           const T      *B,
           TC           *C,
           IndexType    incRowC,
           IndexType    incColC);

//
//  Buffered variant.  Used if the result A^(-1)*B needs to be upcasted for
//  computing C <- A^(-1)*B
//
template <typename T, typename TC, typename IndexType>
    void
    utrlsm(const T     *A,
           const T     *B,
           TC          *C,
           IndexType   incRowC,
           IndexType   incColC);

//
//  Unbuffered variant.
//
template <typename IndexType, typename T>
    void
    utrlsm(const T     *A,
           const T     *B,
           T           *C,
           IndexType   incRowC,
           IndexType   incColC);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_UTRLSM_H

#include <ulmblas/level3/ukernel/utrlsm.tcc>
