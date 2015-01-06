#ifndef ULMBLAS_LEVEL3_PACK_GEPACK_TCC
#define ULMBLAS_LEVEL3_PACK_GEPACK_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/ukernel/ugemm.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename Buffer>
static void
pack_MRxk(IndexType   k,
          bool        conj,
          const TA    *A,
          IndexType   incRowA,
          IndexType   incColA,
          Buffer      *buffer)
{
    const IndexType MR = BlockSizeUGemm<Buffer>::MR;

    if (!conj) {
        for (IndexType j=0; j<k; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                buffer[i] = A[i*incRowA];
            }
            buffer += MR;
            A      += incColA;
        }
    } else {
        for (IndexType j=0; j<k; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                buffer[i] = conjugate(A[i*incRowA]);
            }
            buffer += MR;
            A      += incColA;
        }
    }
}

template <typename IndexType, typename TA, typename Buffer>
void
gepack_A(IndexType   mc,
         IndexType   kc,
         bool        conj,
         const TA    *A,
         IndexType   incRowA,
         IndexType   incColA,
         Buffer      *buffer)
{
    const IndexType MR  = BlockSizeUGemm<Buffer>::MR;
    const IndexType mp  = mc / MR;
    const IndexType mr_ = mc % MR;

    for (IndexType i=0; i<mp; ++i) {
        pack_MRxk(kc, conj, A, incRowA, incColA, buffer);
        buffer += kc*MR;
        A      += MR*incRowA;
    }
    if (mr_>0) {
        for (IndexType j=0; j<kc; ++j) {
            for (IndexType i=0; i<mr_; ++i) {
                buffer[i] = (!conj) ? A[i*incRowA] : conjugate(A[i*incRowA]);
            }
            for (IndexType i=mr_; i<MR; ++i) {
                buffer[i] = Buffer(0);
            }
            buffer += MR;
            A      += incColA;
        }
    }
}

template <typename IndexType, typename TB, typename Buffer>
static void
pack_kxNR(IndexType   k,
          bool        conj,
          const TB    *B,
          IndexType   incRowB,
          IndexType   incColB,
          Buffer      *buffer)
{
    const IndexType NR = BlockSizeUGemm<Buffer>::NR;

    if (!conj) {
        for (IndexType i=0; i<k; ++i) {
            for (IndexType j=0; j<NR; ++j) {
                buffer[j] = B[j*incColB];
            }
            buffer += NR;
            B      += incRowB;
        }
    } else {
        for (IndexType i=0; i<k; ++i) {
            for (IndexType j=0; j<NR; ++j) {
                buffer[j] = conjugate(B[j*incColB]);
            }
            buffer += NR;
            B      += incRowB;
        }
    }
}

template <typename IndexType, typename TB, typename Buffer>
void
gepack_B(IndexType   kc,
         IndexType   nc,
         bool        conj,
         const TB    *B,
         IndexType   incRowB,
         IndexType   incColB,
         Buffer      *buffer)
{
    const IndexType NR  = BlockSizeUGemm<Buffer>::NR;
    const IndexType np  = nc / NR;
    const IndexType nr_ = nc % NR;

    for (IndexType j=0; j<np; ++j) {
        pack_kxNR(kc, conj, B, incRowB, incColB, buffer);
        buffer += kc*NR;
        B      += NR*incColB;
    }
    if (nr_>0) {
        for (IndexType i=0; i<kc; ++i) {
            for (IndexType j=0; j<nr_; ++j) {
                buffer[j] = (!conj) ? B[j*incColB] : conjugate(B[j*incColB]);
            }
            for (IndexType j=nr_; j<NR; ++j) {
                buffer[j] = Buffer(0);
            }
            buffer += NR;
            B      += incRowB;
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_PACK_GEPACK_TCC
