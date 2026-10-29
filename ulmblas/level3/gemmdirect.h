#ifndef ULMBLAS_LEVEL3_GEMMDIRECT_H
#define ULMBLAS_LEVEL3_GEMMDIRECT_H 1

#include <ulmblas/config/simd.h>

namespace ulmBLAS {

//
//  GEMM of small problems directly on the operands, without packing and
//  without buffers (see gemmdirect.tcc).  Returns false if the problem is
//  not handled (then gemm computes it as usual); the general template
//  handles nothing.  Called by gemm after alpha==0 and k==0 are handled.
//
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
    bool
    gemm_direct(IndexType    m,
                IndexType    n,
                IndexType    k,
                const Alpha  &alpha,
                const TA     *A,
                IndexType    incRowA,
                IndexType    incColA,
                const TB     *B,
                IndexType    incRowB,
                IndexType    incColB,
                const Beta   &beta,
                TC           *C,
                IndexType    incRowC,
                IndexType    incColC);

#if defined(USE_NEON)

template <typename IndexType>
    bool
    gemm_direct(IndexType     m,
                IndexType     n,
                IndexType     k,
                const double  &alpha,
                const double  *A,
                IndexType     incRowA,
                IndexType     incColA,
                const double  *B,
                IndexType     incRowB,
                IndexType     incColB,
                const double  &beta,
                double        *C,
                IndexType     incRowC,
                IndexType     incColC);

#endif

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMMDIRECT_H

#include <ulmblas/level3/gemmdirect.tcc>
