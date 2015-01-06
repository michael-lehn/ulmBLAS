#ifndef ULMBLAS_LEVEL3_PACK_HELPACK_H
#define ULMBLAS_LEVEL3_PACK_HELPACK_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename Buffer>
    void
    helpack(IndexType   mc,
            const TA    *A,
            IndexType   incRowA,
            IndexType   incColA,
            Buffer      *buffer);

} // namespace ulmBLAS

#include <ulmblas/level3/pack/helpack.tcc>

#endif // ULMBLAS_LEVEL3_PACK_HELPACK_H
