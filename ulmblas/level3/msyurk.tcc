#ifndef ULMBLAS_LEVEL3_MSYURK_TCC
#define ULMBLAS_LEVEL3_MSYURK_TCC 1

#include <algorithm>
#include <ulmblas/level3/msyurk.h>
#include <ulmblas/level3/ugemm.h>
#include <ulmblas/level3/usyurk.h>

namespace ulmBLAS {

template <typename IndexType, typename T, typename Beta, typename TC>
void
msyurk(IndexType     mc,
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

    assert((MR%NR==0) || (NR%MR==0));

    const IndexType mp = (mc+MR-1) / MR;
    const IndexType np = (nc+NR-1) / NR;

    const IndexType _mr = mc % MR;
    const IndexType _nr = nc % NR;

    const IndexType ki = (MR<NR) ? NR/MR : 1;  // 2
    const IndexType kj = (MR>NR) ? MR/NR : 1;  // 1

    IndexType mr, nr;

    const T *nextA;
    const T *nextB;

    for (IndexType j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;
        nextB = &_B[j*kc*NR];

        for (IndexType i=0; (i/ki)<=(j/kj); ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;
            nextA = &_A[(i+1)*kc*MR];

            if (((i/ki)==(j/kj)) && ((i+1)/ki)>(j/kj)) {
                nextA = _A;
                nextB = &_B[(j+1)*kc*NR];
                if (j==np-1) {
                    nextB = _B;
                }
            }

            if ((i/ki)==(j/kj)) {
                usyurk(mr, nr, kc, i*MR, j*NR,
                       alpha, &_A[i*kc*MR], &_B[j*kc*NR],
                       beta,
                       &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC,
                       nextA, nextB);
            } else {
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
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_SYURK_TCC
