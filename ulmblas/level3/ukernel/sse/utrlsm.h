#ifndef ULMBLAS_LEVEL3_UKERNEL_SSE_UTRLSM_H
#define ULMBLAS_LEVEL3_UKERNEL_SSE_UTRLSM_H 1

namespace ulmBLAS { namespace sse {

template <typename IndexType, typename T>
    void
    utrlsm(const T     *A,
           const T     *B,
           T           *C,
           IndexType   incRowC,
           IndexType   incColC);

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SSE_UTRLSM_H

#include <ulmblas/level3/ukernel/sse/utrlsm.tcc>
