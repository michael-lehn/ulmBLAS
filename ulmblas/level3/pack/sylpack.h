#ifndef ULMBLAS_LEVEL3_PACK_SYLPACK_H
#define ULMBLAS_LEVEL3_PACK_SYLPACK_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename Buffer>
    void
    sylpack(IndexType   mc,
            const TA    *A,
            IndexType   incRowA,
            IndexType   incColA,
            Buffer      *buffer);

} // namespace ulmBLAS

#include <ulmblas/level3/pack/sylpack.tcc>

#endif // ULMBLAS_LEVEL3_PACK_SYLPACK_H
