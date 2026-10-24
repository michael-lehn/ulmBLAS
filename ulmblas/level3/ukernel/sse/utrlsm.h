#ifndef ULMBLAS_LEVEL3_UKERNEL_SSE_UTRLSM_H
#define ULMBLAS_LEVEL3_UKERNEL_SSE_UTRLSM_H 1

#include <ulmblas/level3/ukernel/ref/utrlsm.h>
#include <ulmblas/level3/ukernel/ugemm.h>

namespace ulmBLAS { namespace sse {

//
//  The SSE kernel solves with a lower triangular 4 x 4 block and a
//  4 x 4 right-hand side (hard-coded in the assembly).  The packed panels it
//  receives are built with the block size of the selected GEMM micro kernel.
//
static_assert(ulmBLAS::BlockSizeUGemm<double>::MR==4
           && ulmBLAS::BlockSizeUGemm<double>::NR==4,
              "SSE utrlsm kernel requires MR = NR = 4 for double");

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
