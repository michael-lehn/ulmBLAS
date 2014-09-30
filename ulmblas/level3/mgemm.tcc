#ifndef ULMBLAS_LEVEL3_MGEMM_TCC
#define ULMBLAS_LEVEL3_MGEMM_TCC 1

#include <ulmblas/level3/mgemm.h>
#include <ulmblas/level3/ugemm.h>

namespace ulmBLAS {


template <typename IndexType, typename T, typename Beta, typename TC>
void
mgemm(IndexType     mc,
      IndexType     nc,
      IndexType     kc,
      const T       &alpha,
      const T       *_A,
      const T       *_B,
      const Beta    &beta,
      TC            *C,
      IndexType     incRowC,
      IndexType     incColC)
{
    const IndexType MR = ugemm_mr<T>();
    const IndexType NR = ugemm_nr<T>();

    const IndexType mp = (mc+MR-1) / MR;
    const IndexType np = (nc+NR-1) / NR;

    const IndexType _mr = mc % MR;
    const IndexType _nr = nc % NR;

    IndexType mr, nr;

    const T *nextA;
    const T *nextB;

    for (IndexType j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;
        nextB = &_B[j*kc*NR];

        for (IndexType i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;
            nextA = &_A[(i+1)*kc*MR];

            if (i==mp-1) {
                nextA = _A;
                nextB = &_B[(j+1)*kc*NR];
                if (j==np-1) {
                    nextB = _B;
                }
            }

            if (mr==MR && nr==NR) {
                ugemm(kc,
                      alpha, &_A[i*kc*MR], &_B[j*kc*NR],
                      beta,
                      &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC,
                      nextA, nextB);
            } else {
                // Call the buffered micro kernel
                ugemm(mr, nr, kc,
                      alpha, &_A[i*kc*MR], &_B[j*kc*NR],
                      beta,
                      &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC,
                      nextA, nextB);
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_MGEMM_TCC
