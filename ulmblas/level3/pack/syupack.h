#ifndef ULMBLAS_LEVEL3_PACK_SYUPACK_H
#define ULMBLAS_LEVEL3_PACK_SYUPACK_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename Buffer>
    void
    syupack(IndexType   mc,
            const TA    *A,
            IndexType   incRowA,
            IndexType   incColA,
            Buffer      *buffer);

} // namespace ulmBLAS

#include <ulmblas/level3/pack/syupack.tcc>

#endif // ULMBLAS_LEVEL3_PACK_SYUPACK_H
