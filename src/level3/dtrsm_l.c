#include <ulmblas.h>
#include <level3/dgemm_nn.h>
#include <math.h>

#define MC 16

//
//  Solve A*X = alpha*B
//
//  where A is a lower triangular mxm matrix with unit or non-unit diagonal
//  and B is a general mxn matrix.
//

void
dtrsm_unblk_l(int             unitDiag,
              int             m,
              int             n,
              double          alpha,
              const double    *A,
              int             incRowA,
              int             incColA,
              double          *B,
              int             incRowB,
              int             incColB)
{
    int i, j, k;

//
//  Quick return if possible
//
    if (m==0 || n==0) {
        return;
    }
//
//  And when  alpha equals zero
//
    if (alpha==0.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                B[i*incRowB+j*incColB] = 0.0;
            }
        }
        return;
    }
    for (j=0; j<n; ++j) {
        if (alpha!=1.0) {
            for (i=0; i<m; ++i) {
                B[i*incRowB+j*incColB] *= alpha;
            }
        }
        for (k=0; k<m; ++k) {
            if (B[k*incRowB+j*incColB]!=0.0) {
                if (!unitDiag) {
                    B[k*incRowB+j*incColB] /= A[k*incRowA+k*incColA];
                }
                for (i=k+1; i<m; ++i) {
                    B[i*incRowB+j*incColB] -= B[k*incRowB+j*incColB]
                                             *A[i*incRowA+k*incColA];
                }
            }
        }
    }
}

void
dtrsm_blk_l(int             unitDiag,
            int             m,
            int             n,
            double          alpha,
            const double    *A,
            int             incRowA,
            int             incColA,
            double          *B,
            int             incRowB,
            int             incColB)
{
    int mb  = (m+MC-1) / MC;
    int _mc = m % MC;

    int i, mc;

    for (i=0; i<mb; ++i) {
        mc = (i!=mb-1 || _mc==0) ? MC : _mc;

        dtrsm_unblk_l(unitDiag, mc, n,
                      alpha,
                      &A[i*MC*incRowA+i*MC*incColA], incRowA, incColA,
                      &B[i*MC*incRowB], incRowB, incColB);

        if (i!=mb-1) {
            dgemm_nn(m-(i+1)*MC, n, MC,
                     -1.0,
                     &A[(i+1)*MC*incRowA+i*MC*incColA], incRowA, incColA,
                     &B[ i   *MC*incRowB], incRowB, incColB,
                     alpha,
                     &B[(i+1)*MC*incRowB], incRowB, incColB);
        }

        alpha = 1.0;
    }
}

void
dtrsm_l(int             unitDiag,
        int             m,
        int             n,
        double          alpha,
        const double    *A,
        int             incRowA,
        int             incColA,
        double          *B,
        int             incRowB,
        int             incColB)
{
//
//  Quick return if possible
//
    if (m==0 || n==0) {
        return;
    }

    dtrsm_blk_l(unitDiag, m, n,
                alpha,
                A, incRowA, incColA,
                B, incRowB, incColB);
}
