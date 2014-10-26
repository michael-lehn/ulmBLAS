#ifndef ULMBLAS_LEVEL3_PACK_SYUPACK_TCC
#define ULMBLAS_LEVEL3_PACK_SYUPACK_TCC 1

#include <ulmblas/level3/pack/sylpack.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/ukernel/ugemm.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename Buffer>
static void
syupack_mrxmr(IndexType   mr,
              const TA    *A,
              IndexType   incRowA,
              IndexType   incColA,
              Buffer      *buffer)
{
    const IndexType MR  = ugemm_mr<Buffer>();

    for (IndexType j=0; j<mr; ++j) {
        for (IndexType i=0; i<mr; ++i) {
            buffer[i] = (i<=j) ? A[i*incRowA+j*incColA]
                               : A[j*incRowA+i*incColA];
        }
        for (IndexType i=mr; i<MR; ++i) {
            buffer[i] = Buffer(0);
        }
        buffer += MR;
    }
}

template <typename IndexType, typename TA, typename Buffer>
void
syupack(IndexType   mc,
        const TA    *A,
        IndexType   incRowA,
        IndexType   incColA,
        Buffer      *buffer)
{
    const IndexType MR  = ugemm_mr<Buffer>();
    const IndexType mp  = mc / MR;
    const IndexType mr_ = mc % MR;

    for (IndexType i=0; i<mp; ++i) {
        gepack_A(MR, i*MR,
                 &A[i*MR*incColA], incColA, incRowA,
                 buffer);
        buffer += MR*i*MR;

        syupack_mrxmr(MR,
                      &A[i*MR*incRowA+i*MR*incColA], incRowA, incColA,
                      buffer);
        buffer += MR*MR;

        gepack_A(MR, mc-(i+1)*MR,
                 &A[i*MR*incRowA+(i+1)*MR*incColA], incRowA, incColA,
                 buffer);
        buffer += MR*(mc-(i+1)*MR);
    }

    if (mr_>0) {
        gepack_A(mr_, mc-mr_,
                 &A[mp*MR*incColA], incColA, incRowA,
                 buffer);
        buffer += MR*(mc-mr_);

        syupack_mrxmr(mr_,
                      &A[mp*MR*incRowA+mp*MR*incColA], incRowA, incColA,
                      buffer);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_PACK_SYUPACK_TCC
