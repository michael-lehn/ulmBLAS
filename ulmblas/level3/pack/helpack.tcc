#ifndef ULMBLAS_LEVEL3_PACK_HELPACK_TCC
#define ULMBLAS_LEVEL3_PACK_HELPACK_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level3/pack/helpack.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/ukernel/ugemm.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename Buffer>
static void
helpack_mrxmr(IndexType   mr,
              const TA    *A,
              IndexType   incRowA,
              IndexType   incColA,
              Buffer      *buffer)
{
    const IndexType MR  = BlockSizeUGemm<Buffer>::MR;

    for (IndexType j=0; j<mr; ++j) {
        for (IndexType i=0; i<mr; ++i) {
            buffer[i] = (i>j) ? A[i*incRowA+j*incColA]
                      : (i==j) ? std::real(A[i*incRowA+j*incColA])
                      : conjugate(A[j*incRowA+i*incColA]);
        }
        for (IndexType i=mr; i<MR; ++i) {
            buffer[i] = Buffer(0);
        }
        buffer += MR;
    }
}

template <typename IndexType, typename TA, typename Buffer>
void
helpack(IndexType   mc,
        const TA    *A,
        IndexType   incRowA,
        IndexType   incColA,
        Buffer      *buffer)
{
    const IndexType MR  = BlockSizeUGemm<Buffer>::MR;
    const IndexType mp  = mc / MR;
    const IndexType mr_ = mc % MR;

    for (IndexType i=0; i<mp; ++i) {
        gepack_A(MR, i*MR, false,
                 &A[i*MR*incRowA], incRowA, incColA,
                 buffer);
        buffer += MR*i*MR;

        helpack_mrxmr(MR,
                      &A[i*MR*incRowA+i*MR*incColA], incRowA, incColA,
                      buffer);
        buffer += MR*MR;

        gepack_A(MR, mc-(i+1)*MR, true,
                 &A[i*MR*incColA+(i+1)*MR*incRowA], incColA, incRowA,
                 buffer);
        buffer += MR*(mc-(i+1)*MR);
    }

    if (mr_>0) {
        gepack_A(mr_, mc-mr_, false,
                 &A[mp*MR*incRowA], incRowA, incColA,
                 buffer);
        buffer += MR*(mc-mr_);

        helpack_mrxmr(mr_,
                      &A[mp*MR*incRowA+mp*MR*incColA], incRowA, incColA,
                      buffer);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_PACK_HELPACK_TCC
