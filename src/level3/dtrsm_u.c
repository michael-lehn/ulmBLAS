#include <ulmblas.h>
#include <math.h>
#include <stdio.h>

//
//  Solve A*X = alpha*B
//
//  where A is a upper triangular mxm matrix with unit or non-unit diagonal
//  and B is a general mxn matrix.
//

void
dtrsm_u(enum Diag       diag,
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
        for (k=m-1; k>=0; --k) {
            if (B[k*incRowB+j*incColB]!=0.0) {
                if (diag==NonUnit) {
                    B[k*incRowB+j*incColB] /= A[k*incRowA+k*incColA];
                }
                for (i=0; i<k; ++i) {
                    B[i*incRowB+j*incColB] -= B[k*incRowB+j*incColB]
                                             *A[i*incRowA+k*incColA];
                }
            }
        }
    }
}
