#ifndef ULMBLAS_LEVEL3_PACK_GEPACK_H
#define ULMBLAS_LEVEL3_PACK_GEPACK_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename Buffer>
    void
    gepack_A(IndexType   mc,
             IndexType   kc,
             const TA    *A,
             IndexType   incRowA,
             IndexType   incColA,
             Buffer      *buffer);

template <typename IndexType, typename TB, typename Buffer>
    void
    gepack_B(IndexType   kc,
             IndexType   nc,
             const TB    *B,
             IndexType   incRowB,
             IndexType   incColB,
             Buffer      *buffer);

} // namespace ulmBLAS

#include <ulmblas/level3/pack/gepack.tcc>

#endif // ULMBLAS_LEVEL3_PACK_GEPACK_H
