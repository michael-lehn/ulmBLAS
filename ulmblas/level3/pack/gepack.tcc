#ifndef ULMBLAS_LEVEL3_PACK_GEPACK_TCC
#define ULMBLAS_LEVEL3_PACK_GEPACK_TCC 1

#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/ugemm.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename Buffer>
static void
pack_MRxk(IndexType   k,
          const TA    *A,
          IndexType   incRowA,
          IndexType   incColA,
          Buffer      *buffer)
{
    const IndexType MR = ugemm_mr<Buffer>();

    for (IndexType j=0; j<k; ++j) {
        for (IndexType i=0; i<MR; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += MR;
        A      += incColA;
    }
}

template <typename IndexType, typename TA, typename Buffer>
void
gepack_A(IndexType   mc,
         IndexType   kc,
         const TA    *A,
         IndexType   incRowA,
         IndexType   incColA,
         Buffer      *buffer)
{
    const IndexType MR  = ugemm_mr<Buffer>();
    const IndexType mp  = mc / MR;
    const IndexType _mr = mc % MR;

    for (IndexType i=0; i<mp; ++i) {
        pack_MRxk(kc, A, incRowA, incColA, buffer);
        buffer += kc*MR;
        A      += MR*incRowA;
    }
    if (_mr>0) {
        for (IndexType j=0; j<kc; ++j) {
            for (IndexType i=0; i<_mr; ++i) {
                buffer[i] = A[i*incRowA];
            }
            for (IndexType i=_mr; i<MR; ++i) {
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
          const TB    *B,
          IndexType   incRowB,
          IndexType   incColB,
          Buffer      *buffer)
{
    const IndexType NR = ugemm_nr<Buffer>();

    for (IndexType i=0; i<k; ++i) {
        for (IndexType j=0; j<NR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += NR;
        B      += incRowB;
    }
}

template <typename IndexType, typename TB, typename Buffer>
void
gepack_B(IndexType   kc,
         IndexType   nc,
         const TB    *B,
         IndexType   incRowB,
         IndexType   incColB,
         Buffer      *buffer)
{
    const IndexType NR  = ugemm_nr<Buffer>();
    const IndexType np  = nc / NR;
    const IndexType _nr = nc % NR;

    for (IndexType j=0; j<np; ++j) {
        pack_kxNR(kc, B, incRowB, incColB, buffer);
        buffer += kc*NR;
        B      += NR*incColB;
    }
    if (_nr>0) {
        for (IndexType i=0; i<kc; ++i) {
            for (IndexType j=0; j<_nr; ++j) {
                buffer[j] = B[j*incColB];
            }
            for (IndexType j=_nr; j<NR; ++j) {
                buffer[j] = Buffer(0);
            }
            buffer += NR;
            B      += incRowB;
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_PACK_GEPACK_TCC
