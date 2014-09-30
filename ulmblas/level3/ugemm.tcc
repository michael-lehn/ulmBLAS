#ifndef ULMBLAS_LEVEL3_UGEMM_TCC
#define ULMBLAS_LEVEL3_UGEMM_TCC 1

#include <ulmblas/level1extensions/geaxpy.h>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/ugemm.h>
#include <ulmblas/level3/kernel/ugemm.h>

namespace ulmBLAS {

template <typename T>
int
ugemm_mr()
{
    return SELECT_UGEMM_KERNEL::ugemm_mr<T>();
}

template <typename T>
int
ugemm_nr()
{
    return SELECT_UGEMM_KERNEL::ugemm_nr<T>();
}

//
//  Buffered variant.  Used for zero padded panels.
//
template <typename IndexType, typename T, typename Beta, typename TC>
void
ugemm(IndexType    mr,
      IndexType    nr,
      IndexType    kc,
      const T      &alpha,
      const T      *A,
      const T      *B,
      const Beta   &beta,
      TC           *C,
      IndexType    incRowC,
      IndexType    incColC,
      const T      *nextA,
      const T      *nextB)
{
    const IndexType MR = ugemm_mr<T>();
    const IndexType NR = ugemm_nr<T>();

    T   _C[MR*NR];

    ugemm(kc, alpha, A, B, T(0), _C, IndexType(1), MR, nextA, nextB);
    gescal(mr, nr, beta, C, incRowC, incColC);
    geaxpy(mr, nr, Beta(1), _C, IndexType(1), MR, C, incRowC, incColC);
}

//
//  Buffered variant.  Used if the result alpha*A*B needs to be upcasted for
//  computing C <- beta*C + (alpha*A*B)
//
template <typename IndexType, typename T, typename Beta, typename TC>
void
ugemm(IndexType   kc,
      const T     &alpha,
      const T     *A,
      const T     *B,
      const Beta  &beta,
      TC          *C,
      IndexType   incRowC,
      IndexType   incColC,
      const T     *nextA,
      const T     *nextB)
{
    const IndexType MR = ugemm_mr<T>();
    const IndexType NR = ugemm_nr<T>();

    ugemm(MR, NR, kc, alpha, A, B, beta, C, incRowC, incColC, nextA, nextB);
}

//
//  Unbuffered variant.
//
template <typename IndexType, typename T>
void
ugemm(IndexType   kc,
      const T     &alpha,
      const T     *A,
      const T     *B,
      const T     &beta,
      T           *C,
      IndexType   incRowC,
      IndexType   incColC,
      const T     *nextA,
      const T     *nextB)
{
    SELECT_UGEMM_KERNEL::ugemm(kc, alpha, A, B, beta, C, incRowC, incColC,
                               nextA, nextB);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UGEMM_TCC
