#ifndef ULMBLAS_LEVEL3_MTRUMM_TCC
#define ULMBLAS_LEVEL3_MTRUMM_TCC 1

#include <ulmblas/level3/mtrumm.tcc>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename T, typename TB>
void
mtrumm(IndexType    mc,
       IndexType    nc,
       const Alpha  &alpha,
       const T      *_A,
       const T      *_B,
       TB           *B,
       IndexType    incRowB,
       IndexType    incColB)
{
    const IndexType MR = ugemm_mr<T>();
    const IndexType NR = ugemm_nr<T>();

    const IndexType mp = (mc+MR-1) / MR;
    const IndexType np = (nc+NR-1) / NR;

    const IndexType _mr = mc % MR;
    const IndexType _nr = nc % NR;

    IndexType mr, nr;
    IndexType kc;

    const T Zero(0);

    const T *nextA;
    const T *nextB;

    for (IndexType j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;
        nextB = &_B[j*mc*NR];


        IndexType ia = 0;
        for (IndexType i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;
            kc    = mc - i*MR;
            nextA = &_A[(ia+1)*MR];

            if (i==mp-1) {
                nextA = _A;
                nextB = &_B[(j+1)*mc*NR];
                if (j==np-1) {
                    nextB = _B;
                }
            }

            if (mr==MR && nr==NR) {
                ugemm(kc,
                      alpha, &_A[ia*MR], &_B[j*mc*NR+i*MR*NR],
                      Zero,
                      &B[i*MR*incRowB+j*NR*incColB], incRowB, incColB,
                      nextA, nextB);
            } else {
                // Call the buffered micro kernel
                ugemm(mr, nr, kc,
                      alpha, &_A[ia*MR], &_B[j*mc*NR+i*MR*NR],
                      Zero,
                      &B[i*MR*incRowB+j*NR*incColB], incRowB, incColB,
                      nextA, nextB);
            }
            ia += kc;
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_MTRUMM_TCC
