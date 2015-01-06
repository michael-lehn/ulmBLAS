#ifndef ULMBLAS_LEVEL3_UKERNEL_SSE_UTRLSM_H
#define ULMBLAS_LEVEL3_UKERNEL_SSE_UTRLSM_H 1

#include <ulmblas/level3/ukernel/ref/utrlsm.h>

namespace ulmBLAS { namespace sse {

using ref::utrlsm;

template <typename IndexType>
static typename std::enable_if<std::is_convertible<IndexType,long>::value,
    void>::type
    utrlsm(const double  *A,
           const double  *B,
           double        *C,
           IndexType     incRowC,
           IndexType     incColC);

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SSE_UTRLSM_H

#include <ulmblas/level3/ukernel/sse/utrlsm.tcc>
