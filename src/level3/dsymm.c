#include <ulmblas.h>
#include <auxiliary/xerbla.h>
#include <math.h>
void
ULMBLAS(dsymm)(const enum Side   side,
               const enum UpLo   upLo,
               const int         m,
               const int         n,
               const double      alpha,
               const double      *A,
               const int         ldA,
               const double      *B,
               const int         ldB,
               const double      beta,
               double            *C,
               const int         ldC)
{
//
//  Local scalars
//
    int     i, j, k;
    double  tmp1, tmp2;

//
//  Quick return if possible.
//
    if (m==0 || n==0 || (alpha==0.0 && beta==1.0)) {
        return;
    }

//
//  And when alpha is exactly zero
//
    if (alpha==0.0) {
        if (beta==0.0) {
            for (j=0; j<n; ++j) {
                for (i=0; i<m; ++i) {
                    C[i+j*ldC] = 0.0;
                }
            }
        } else {
            for (j=0; j<n; ++j) {
                for (i=0; i<m; ++i) {
                    C[i+j*ldC] *= beta;
                }
            }
        }
        return;
    }
//
//  Start the operations.
//
    if (side==Left) {
//
//      Form  C := alpha*A*B + beta*C.
//
        if (upLo==Upper) {
            for (j=0; j<n; ++j) {
                for (i=0; i<m; ++i) {
                    tmp1 = alpha*B[i+j*ldB];
                    tmp2 = 0.0;
                    for (k=0; k<i; ++k) {
                        C[k+j*ldC] += tmp1*A[k+i*ldA];
                        tmp2       += B[k+j*ldB]*A[k+i*ldA];
                    }
                    if (beta==0.0) {
                        C[i+j*ldC] = tmp1*A[i+i*ldA] + alpha*tmp2;
                    } else {
                        C[i+j*ldC] = beta*C[i+j*ldC]
                                   + tmp1*A[i+i*ldA]
                                   + alpha*tmp2;
                    }
                }
            }
        } else {
            for (j=0; j<n; ++j) {
                for (i=m-1; i>=0; --i) {
                    tmp1 = alpha*B[i+j*ldB];
                    tmp2 = 0.0;
                    for (k=i+1; k<m; ++k) {
                        C[k+j*ldC] += tmp1*A[k+i*ldA];
                        tmp2       += B[k+j*ldB]*A[k+i*ldA];
                    }
                    if (beta==0.0) {
                        C[i+j*ldC] = tmp1*A[i+i*ldA] + alpha*tmp2;
                    } else {
                        C[i+j*ldC] = beta*C[i+j*ldC]
                                   + tmp1*A[i+i*ldA]
                                   + alpha*tmp2;
                    }
                 }
            }
        }
    } else {
//
//      Form  C := alpha*B*A + beta*C.
//
        for (j=0; j<n; ++j) {
            tmp1 = alpha*A[j+j*ldA];
            if (beta==0.0) {
                for (i=0; i<m; ++i) {
                    C[i+j*ldC] = tmp1*B[i+j*ldB];
                }
            } else {
                for (i=0; i<m; ++i) {
                    C[i+j*ldC] = beta*C[i+j*ldC] + tmp1*B[i+j*ldB];
                }
            }
            for (k=0; k<j; ++k) {
                if (upLo==Upper) {
                    tmp1 = alpha*A[k+j*ldA];
                } else {
                    tmp1 = alpha*A[j+k*ldA];
                }
                for (i=0; i<m; ++i) {
                    C[i+j*ldC] += tmp1*B[i+k*ldB];
                }
            }
            for (k=j+1; k<n; ++k) {
                if (upLo==Upper) {
                    tmp1 = alpha*A[j+k*ldA];
                } else {
                    tmp1 = alpha*A[k+j*ldA];
                }
                for (i=0; i<m; ++i) {
                    C[i+j*ldC] += tmp1*B[i+k*ldB];
                }
            }
        }
    }
}

void
F77BLAS(dsymm)(const char     *_side,
               const char     *_upLo,
               const int      *_m,
               const int      *_n,
               const double   *_alpha,
               const double   *A,
               const int      *_ldA,
               const double   *B,
               const int      *_ldB,
               const double   *_beta,
               double         *C,
               const int      *_ldC)
{
//
//  Dereference scalar parameters
//
    enum Side side = charToSide(*_side);
    enum UpLo upLo = charToUpLo(*_upLo);
    int m          = *_m;
    int n          = *_n;
    double alpha   = *_alpha;
    int ldA        = *_ldA;
    int ldB        = *_ldB;
    double beta    = *_beta;
    int ldC        = *_ldC;

//
//  Set  numRowsA as the number of rows of A
//
    int numRowsA = (side==Left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;
    if (side==0) {
        info = 1;
    } else if (upLo==0) {
        info = 2;
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (ldA<max(1,numRowsA)) {
        info = 7;
    } else if (ldB<m) {
        info = 9;
    } else if (ldC<m) {
        info = 12;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DSYMM ", &info);
    }

    ULMBLAS(dsymm)(side, upLo,
                   m, n,
                   alpha,
                   A, ldA,
                   B, ldB,
                   beta,
                   C, ldC);
}
