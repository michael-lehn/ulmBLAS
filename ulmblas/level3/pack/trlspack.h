#ifndef ULMBLAS_LEVEL3_PACK_TRLSPACK_H
#define ULMBLAS_LEVEL3_PACK_TRLSPACK_H 1

namespace ulmBLAS {

template <typename IndexType, typename TL, typename Buffer>
    void
    trlspack(IndexType   mc,
             bool        conjA,
             bool        unit,
             const TL    *L,
             IndexType   incRowL,
             IndexType   incColL,
             Buffer      *buffer);

} // namespace ulmBLAS

#include <ulmblas/level3/pack/trlspack.tcc>

#endif // ULMBLAS_LEVEL3_PACK_TRLSPACK_H
