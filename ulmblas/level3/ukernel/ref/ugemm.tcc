#ifndef ULMBLAS_LEVEL3_UKERNEL_REF_UGEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_REF_UGEMM_TCC 1

#include <ulmblas/config/blocksize.h>
#include <ulmblas/level3/ukernel/ref/ugemm.h>

namespace ulmBLAS { namespace ref {

template <typename T>
int
ugemm_mr()
{
    return BlockSize<T>::MR;
}

template <typename T>
int
ugemm_nr()
{
    return BlockSize<T>::NR;
}

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
    const IndexType UnitStride = 1;

    const IndexType MR = ugemm_mr<T>();
    const IndexType NR = ugemm_nr<T>();

    T  C_[MR*NR];

    ugemm(kc, alpha, A, B, T(0), C_, UnitStride, MR, nextA, nextB);
    gescal(MR, NR, beta, C, incRowC, incColC);
    geaxpy(MR, NR, Beta(1), C_, UnitStride, MR, C, incRowC, incColC);
}

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
    const IndexType MR = ugemm_mr<T>();
    const IndexType NR = ugemm_nr<T>();

    T AB[MR*NR];

    for (IndexType i=0; i<MR*NR; ++i) {
        AB[i] = T(0);
    }

    for (IndexType l=0; l<kc; ++l) {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                AB[i+j*MR] += A[i]*B[j];
            }
        }
        A += MR;
        B += NR;
    }

    if (beta==0.0) {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] = T(0);
            }
        }
    } else {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] *= beta;
            }
        }
    }

    if (alpha==1.0) {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += AB[i+j*MR];
            }
        }
    } else {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += alpha*AB[i+j*MR];
            }
        }
    }
}

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_REF_UGEMM_TCC 1
