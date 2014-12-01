#ifndef ULMBLAS_LEVEL3_UKERNEL_REF_UTRLSM_H
#define ULMBLAS_LEVEL3_UKERNEL_REF_UTRLSM_H 1

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename T>
    void
    utrlsm(const T     *A,
           const T     *B,
           T           *C,
           IndexType   incRowC,
           IndexType   incColC);

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_REF_UTRLSM_H

#include <ulmblas/level3/ukernel/ref/utrlsm.tcc>
