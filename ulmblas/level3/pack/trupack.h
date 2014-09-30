#ifndef ULMBLAS_LEVEL3_PACK_TRUPACK_H
#define ULMBLAS_LEVEL3_PACK_TRUPACK_H 1

namespace ulmBLAS {

template <typename IndexType, typename TU, typename Buffer>
    void
    trupack(IndexType   mc,
            bool        unit,
            const TU    *U,
            IndexType   incRowU,
            IndexType   incColU,
            Buffer      *buffer);

} // namespace ulmBLAS

#include <ulmblas/level3/pack/trupack.tcc>

#endif // ULMBLAS_LEVEL3_PACK_TRUPACK_H
