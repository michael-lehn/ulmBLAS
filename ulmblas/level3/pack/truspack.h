#ifndef ULMBLAS_LEVEL3_PACK_TRUSPACK_H
#define ULMBLAS_LEVEL3_PACK_TRUSPACK_H 1

namespace ulmBLAS {

template <typename IndexType, typename TU, typename Buffer>
    void
    truspack(IndexType   mc,
             bool        unit,
             const TU    *U,
             IndexType   incRowU,
             IndexType   incColU,
             Buffer      *buffer);

} // namespace ulmBLAS

#include <ulmblas/level3/pack/truspack.tcc>

#endif // ULMBLAS_LEVEL3_PACK_TRUSPACK_H
