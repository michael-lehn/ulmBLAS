#ifndef ULMBLAS_LEVEL3_PACK_TRLPACK_H
#define ULMBLAS_LEVEL3_PACK_TRLPACK_H 1

namespace ulmBLAS {

template <typename IndexType, typename TL, typename Buffer>
    void
    trlpack(IndexType   mc,
            bool        conj,
            bool        unit,
            const TL    *L,
            IndexType   incRowL,
            IndexType   incColL,
            Buffer      *buffer);

} // namespace ulmBLAS

#include <ulmblas/level3/pack/trlpack.tcc>

#endif // ULMBLAS_LEVEL3_PACK_TRLPACK_H
