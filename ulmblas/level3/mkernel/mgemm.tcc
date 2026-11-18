#ifndef ULMBLAS_LEVEL3_MKERNEL_MGEMM_TCC
#define ULMBLAS_LEVEL3_MKERNEL_MGEMM_TCC 1

#include <ulmblas/level3/gemm/macrokernel.h>
#include <ulmblas/level3/mkernel/mgemm.h>
#include <ulmblas/level3/ukernel/backend.h>

namespace ulmBLAS {

template <typename IndexType, typename T, typename Beta, typename TC>
void
mgemm(IndexType mc, IndexType nc, IndexType kc, const T &alpha, const T *A_,
      const T *B_, const Beta &beta, TC *C, IndexType incRowC,
      IndexType incColC)
{
    MacroKernel<DefaultBackend<T>>::run(mc, nc, kc, alpha, A_, B_, beta, C,
                                        incRowC, incColC);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_MKERNEL_MGEMM_TCC
