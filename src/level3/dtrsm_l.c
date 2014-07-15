#include <ulmblas.h>
#include <level3/dgemm_nn.h>
#include <math.h>

#define MC 128

//
//  Solve A*X = alpha*B
//
//  where A is a lower triangular mxm matrix with unit or non-unit diagonal
//  and B is a general mxn matrix.
//

static void
dtrsm_unblk_l(enum Diag       diag,
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

    for (j=0; j<n; ++j) {
        if (alpha!=1.0) {
            for (i=0; i<m; ++i) {
                B[i*incRowB+j*incColB] *= alpha;
            }
        }
        for (k=0; k<m; ++k) {
            if (B[k*incRowB+j*incColB]!=0.0) {
                if (diag==NonUnit) {
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
dtrsm_l(enum Diag       diag,
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
    int mb  = (m+MC-1)/MC;
    int _mc = m % MC;

    int mc, i, j;

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

    for (j=0; j<mb; ++j) {
        mc = (j!=mb-1 || _mc==0) ? MC : _mc;

        dtrsm_unblk_l(diag, mc, n, alpha,
                      &A[j*MC*incRowA+j*MC*incColA], incRowA, incColA,
                      &B[j*MC*incRowB], incRowB, incColB);

        if (m-MC*(j+1)>0) {
            dgemm_nn(m-MC*(j+1), n, mc, -1.0,
                     &A[(j+1)*MC*incRowA+j*MC*incColA], incRowA, incColB,
                     &B[j*MC*incRowB], incRowB, incColB,
                     1.0,
                     &B[(j+1)*MC*incRowB], incRowB, incColB);
        }
    }
}
