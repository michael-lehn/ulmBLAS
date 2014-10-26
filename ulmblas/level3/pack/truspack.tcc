#ifndef ULMBLAS_LEVEL3_PACK_TRUSPACK_TCC
#define ULMBLAS_LEVEL3_PACK_TRUSPACK_TCC 1

#include <ulmblas/level3/pack/trlpack.h>
#include <ulmblas/level3/ukernel/ugemm.h>

namespace ulmBLAS {

template <typename IndexType, typename TU, typename Buffer>
static void
truspack_MRxk(IndexType   k,
              bool        unit,
              const TU    *U,
              IndexType   incRowU,
              IndexType   incColU,
              Buffer      *buffer)
{
    const IndexType MR  = ugemm_mr<Buffer>();

    for (IndexType j=0; j<MR; ++j) {
        for (IndexType i=0; i<j; ++i) {
            buffer[i] = U[i*incRowU];
        }
        buffer[j] = (unit) ? Buffer(1) : Buffer(1)/U[j*incRowU];
        for (IndexType i=j+1; i<MR; ++i) {
            buffer[i] = Buffer(0);
        }
        buffer += MR;
        U      += incColU;
    }
    for (IndexType j=0; j<k-MR; ++j) {
        for (IndexType i=0; i<MR; ++i) {
            buffer[i] = U[i*incRowU];
        }
        buffer += MR;
        U      += incColU;
    }
}

template <typename IndexType, typename TU, typename Buffer>
void
truspack(IndexType   mc,
         bool        unit,
         const TU    *U,
         IndexType   incRowU,
         IndexType   incColU,
         Buffer      *buffer)
{
    const IndexType MR  = ugemm_mr<Buffer>();
    const IndexType mp  = mc / MR;
    const IndexType mr_ = mc % MR;

    for (IndexType i=0; i<mp; ++i) {
        truspack_MRxk(mc-i*MR, unit, U, incRowU, incColU, buffer);
        buffer += (mc-i*MR)*MR;
        U      += MR*(incRowU+incColU);
    }

    if (mr_>0) {
        for (IndexType j=0; j<mr_; ++j) {
            for (IndexType i=0; i<j; ++i) {
                buffer[i] = U[i*incRowU];
            }
            buffer[j] = (unit) ? Buffer(1) : Buffer(1)/U[j*incRowU];
            for (IndexType i=j+1; i<MR; ++i) {
                buffer[i] = Buffer(0);
            }
            buffer += MR;
            U      += incColU;
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_PACK_TRUSPACK_TCC
