#ifndef ULMBLAS_LEVEL3_PACK_TRLPACK_TCC
#define ULMBLAS_LEVEL3_PACK_TRLPACK_TCC 1

#include <ulmblas/level3/pack/trlpack.h>
#include <ulmblas/level3/ugemm.h>

namespace ulmBLAS {

template <typename IndexType, typename TL, typename Buffer>
static void
trlpack_MRxk(IndexType   k,
             bool        unit,
             const TL    *L,
             IndexType   incRowL,
             IndexType   incColL,
             Buffer      *buffer)
{
    const IndexType MR  = ugemm_mr<Buffer>();

    for (IndexType j=0; j<k-MR; ++j) {
        for (IndexType i=0; i<MR; ++i) {
            buffer[i] = L[i*incRowL];
        }
        buffer += MR;
        L      += incColL;
    }
    for (IndexType j=0; j<MR; ++j) {
        for (IndexType i=0; i<j; ++i) {
            buffer[i] = Buffer(0);
        }
        buffer[j] = (unit) ? Buffer(1) : L[j*incRowL];
        for (IndexType i=j+1; i<MR; ++i) {
            buffer[i] = L[i*incRowL];
        }
        buffer += MR;
        L      += incColL;
    }
}

template <typename IndexType, typename TL, typename Buffer>
void
trlpack(IndexType   mc,
        bool        unit,
        const TL    *L,
        IndexType   incRowL,
        IndexType   incColL,
        Buffer      *buffer)
{
    const IndexType MR  = ugemm_mr<Buffer>();
    const IndexType mp  = mc / MR;
    const IndexType _mr = mc % MR;

    for (IndexType i=0; i<mp; ++i) {
        trlpack_MRxk((i+1)*MR, unit, L, incRowL, incColL, buffer);
        buffer += (i+1)*MR*MR;
        L      += MR*incRowL;
    }

    if (_mr>0) {
        for (IndexType j=0; j<mp*MR; ++j) {
            for (IndexType i=0; i<_mr; ++i) {
                buffer[i] = L[i*incRowL];
            }
            for (IndexType i=_mr; i<MR; ++i) {
                buffer[i] = Buffer(0);
            }
            buffer += MR;
            L      += incColL;
        }
        for (IndexType j=0; j<_mr; ++j) {
            for (IndexType i=0; i<j; ++i) {
                buffer[i] = Buffer(0);
            }
            buffer[j] = (unit) ? Buffer(1) : L[j*incRowL];
            for (IndexType i=j+1; i<_mr; ++i) {
                buffer[i] = L[i*incRowL];
            }
            for (IndexType i=_mr; i<MR; ++i) {
                buffer[i] = Buffer(0);
            }
            buffer += MR;
            L      += incColL;
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_PACK_TRLPACK_TCC
