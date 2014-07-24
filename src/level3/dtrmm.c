#include <ulmblas.h>
#include <auxiliary/xerbla.h>
#include <level3/dgemm_nn.h>

void
ULMBLAS(dtrmm)(enum Side      side,
               enum UpLo      upLo,
               enum Trans     transA,
               enum Diag      diag,
               int            m,
               int            n,
               double         alpha,
               const double   *A,
               int            ldA,
               double         *B,
               int            ldB)
{
//
//  Local scalars
//
    int     i, j, k;
    double  tmp;

//
//  Quick return if possible.
//
    if (m==0 || n==0) {
        return;
    }

//
//  And if alpha is exactly zero
//
    if (alpha==0.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                B[i+j*ldB] = 0.0;
            }
        }
    }

    if (side==Left) {
        if (transA==NoTrans) {
            if (upLo==Upper) {
                for (j=0; j<n; ++j) {
                    for (k=0; k<m; ++k) {
                        tmp = alpha*B[k+j*ldB];
                        for (i=0; i<k; ++i) {
                            B[i+j*ldB] += tmp*A[i+k*ldA];
                        }
                        if (diag==NonUnit) {
                            tmp *= A[k+k*ldA];
                        }
                        B[k+j*ldB] = tmp;
                    }
                }
            } else {
                for (j=0; j<n; ++j) {
                    for (k=m-1; k>=0; --k) {
                        tmp = alpha*B[k+j*ldB];
                        B[k+j*ldB] = tmp;
                        if (diag==NonUnit) {
                            B[k+j*ldB] *= A[k+k*ldA];
                        }
                        for (i=k+1; i<m; ++i) {
                            B[i+j*ldB] += tmp*A[i+k*ldA];
                        }
                    }
                }
            }
        } else {
            if (upLo==Upper) {
                for (j=0; j<n; ++j) {
                    for (i=m-1; i>=0; --i) {
                        tmp = B[i+j*ldB];
                        if (diag==NonUnit) {
                            tmp *= A[i+i*ldA];
                        }
                        for (k=0; k<i; ++k) {
                            tmp += A[k+i*ldA] * B[k+j*ldB];
                        }
                        B[i+j*ldB] = alpha*tmp;
                    }
                }
            } else {
                for (j=0; j<n; ++j) {
                    for (i=0; i<m; ++i) {
                        tmp = B[i+j*ldB];
                        if (diag==NonUnit) {
                            tmp *= A[i+i*ldA];
                        }
                        for (k=i+1; k<m; ++k) {
                            tmp += A[k+i*ldA] * B[k+j*ldB];
                        }
                        B[i+j*ldB] = alpha*tmp;
                    }
                }
            }
        }
    } else {
        if (transA==NoTrans) {
            if (upLo==Upper) {
                for (j=n-1; j>=0; --j) {
                    tmp = alpha;
                    if (diag==NonUnit) {
                        tmp *= A[j+j*ldA];
                    }
                    for (i=0; i<m; ++i) {
                        B[i+j*ldB] *= tmp;
                    }
                    for (k=0; k<j; ++k) {
                        if (A[k+j*ldA]!=0.0) {
                            tmp = alpha*A[k+j*ldA];
                            for (i=0; i<m; ++i) {
                                B[i+j*ldB] += tmp*B[i+k*ldB];
                            }
                        }
                    }
                }
            } else {
                for (j=0; j<n; ++j) {
                    tmp = alpha;
                    if (diag==NonUnit) {
                        tmp *= A[j+j*ldA];
                    }
                    for (i=0; i<m; ++i) {
                        B[i+j*ldB] *= tmp;
                    }
                    for (k=j+1; k<n; ++k) {
                        if (A[k+j*ldA]!=0.0) {
                            tmp = alpha*A[k+j*ldA];
                            for (i=0; i<m; ++i) {
                                B[i+j*ldB] += tmp*B[i+k*ldB];
                            }
                        }
                    }
                }
            }
        } else {
            if (upLo==Upper) {
                for (k=0; k<n; ++k) {
                    for (j=0; j<k; ++j) {
                        if (A[j+k*ldA]!=0.0) {
                            tmp = alpha*A[j+k*ldA];
                            for (i=0; i<m; ++i) {
                                B[i+j*ldB] += tmp*B[i+k*ldB];
                            }
                        }
                    }
                    tmp = alpha;
                    if (diag==NonUnit) {
                        tmp *= A[k+k*ldA];
                    }
                    if (tmp!=1.0) {
                        for (i=0; i<m; ++i) {
                            B[i+k*ldB] *= tmp;
                        }
                    }
                }
            } else {
                for (k=n-1; k>=0; --k) {
                    for (j=k+1; j<n; ++j) {
                        if (A[j+k*ldA]!=0.0) {
                            tmp = alpha*A[j+k*ldA];
                            for (i=0; i<m; ++i) {
                                B[i+j*ldB] += tmp*B[i+k*ldB];
                            }
                        }
                    }
                    tmp = alpha;
                    if (diag==NonUnit) {
                        tmp *= A[k+k*ldA];
                    }
                    if (tmp!=1.0) {
                        for (i=0; i<m; ++i) {
                            B[i+k*ldB] *= tmp;
                        }
                    }
                }
            }
        }
    }
}

void
F77BLAS(dtrmm)(const char     *_side,
               const char     *_upLo,
               const char     *_transA,
               const char     *_diag,
               const int      *_m,
               const int      *_n,
               const double   *_alpha,
               const double   *A,
               const int      *_ldA,
               double         *B,
               const int      *_ldB)

{
//
//  Dereference scalar parameters
//
    enum Side  side   = charToSide(*_side);
    enum UpLo  upLo   = charToUpLo(*_upLo);
    enum Trans transA = charToTranspose(*_transA);
    enum Diag  diag   = charToDiag(*_diag);
    int m             = *_m;
    int n             = *_n;
    double alpha      = *_alpha;
    int ldA           = *_ldA;
    int ldB           = *_ldB;


    int numRowsA = (side==Left) ? m : n;
//
//  Test the input parameters
//
    int info = 0;
    if (side==0) {
        info = 1;
    } else if (upLo==0) {
        info = 2;
    } else if (transA==0) {
        info = 3;
    } else if (diag==0) {
        info = 4;
    } else if (m<0) {
        info = 5;
    } else if (n<0) {
        info = 6;
    } else if (ldA<max(1,numRowsA)) {
        info = 9;
    } else if (ldB<max(1,m)) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DTRMM ", &info);
    }

    ULMBLAS(dtrmm)(side, upLo, transA, diag,
                   m, n,
                   alpha,
                   A, ldA,
                   B, ldB);
}
