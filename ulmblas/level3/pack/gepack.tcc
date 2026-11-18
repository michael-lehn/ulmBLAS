#ifndef ULMBLAS_LEVEL3_PACK_GEPACK_TCC
#define ULMBLAS_LEVEL3_PACK_GEPACK_TCC 1

#include <ulmblas/level3/gemm/pack.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/ukernel/backend.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename Buffer>
void
gepack_A(IndexType mc, IndexType kc, bool conj, const TA *A, IndexType incRowA,
         IndexType incColA, Buffer *buffer)
{
    PackA<DefaultBackend<Buffer>>::run(mc, kc, conj, A, incRowA, incColA,
                                       buffer);
}

template <typename IndexType, typename TB, typename Buffer>
void
gepack_B(IndexType kc, IndexType nc, bool conj, const TB *B, IndexType incRowB,
         IndexType incColB, Buffer *buffer)
{
    PackB<DefaultBackend<Buffer>>::run(kc, nc, conj, B, incRowB, incColB,
                                       buffer);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_PACK_GEPACK_TCC
