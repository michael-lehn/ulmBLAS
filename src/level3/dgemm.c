#include <ulmblas.h>
#include <auxiliary/xerbla.h>
#include <level3/dgemm_nn.h>

void
ULMBLAS(dgemm)(const enum Trans  transA,
               const enum Trans  transB,
               const int         m,
               const int         n,
               const int         k,
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
    int     i, j;

#if defined(ULM_REFERENCE)
    int     l;
    double  tmp;
#endif

//
//  Quick return if possible.
//
    if (m==0 || n==0 || ((alpha==0.0 || k==0) && (beta==1.0))) {
        return;
    }

//
//  And if alpha is exactly zero
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
    if (transB==NoTrans || transB==Conj) {
        if (transA==NoTrans || transA==Conj) {
//
//          Form  C := alpha*A*B + beta*C.
//
#if defined(ULM_REFERENCE)
            for (j=0; j<n; ++j) {
                if (beta==0.0) {
                    for (i=0; i<m; ++i) {
                        C[i+j*ldC] = 0.0;
                    }
                } else if (beta!=1.0) {
                    for (i=0; i<m; ++i) {
                        C[i+j*ldC] *= beta;
                    }
                }
                for (l=0; l<k; ++l) {
                    if (B[l+j*ldB]!=0.0) {
                        tmp = alpha*B[l+j*ldB];
                        for (i=0; i<m; ++i) {
                            C[i+j*ldC] += tmp*A[i+l*ldA];
                        }
                    }
                }
            }
#elif defined(ULM_BLOCKED)
            dgemm_nn(m, n, k,
                     alpha,
                     A, 1, ldA,
                     B, 1, ldB,
                     beta,
                     C, 1, ldC);
#else
#error      "no implementation specified!\n"
#endif
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
#if defined(ULM_REFERENCE)
            for (j=0; j<n; ++j) {
                for (i=0; i<m; ++i) {
                    tmp = 0.0;
                    for (l=0; l<k; ++l) {
                        tmp += A[l+i*ldA]*B[l+j*ldB];
                    }
                    if (beta==0.0) {
                        C[i+j*ldC] = alpha*tmp;
                    } else {
                        C[i+j*ldC] = alpha*tmp + beta*C[i+j*ldC];
                    }
                }
            }
#elif defined(ULM_BLOCKED)
            dgemm_nn(m, n, k,
                     alpha,
                     A, ldA, 1,
                     B, 1, ldB,
                     beta,
                     C, 1, ldC);
#else
#error      "no implementation specified!\n"
#endif
        }
    } else {
        if (transA==NoTrans || transA==Conj) {
//
//          Form  C := alpha*A*B**T + beta*C
//
#if defined(ULM_REFERENCE)
            for (j=0; j<n; ++j) {
                if (beta==0.0) {
                    for (i=0; i<m; ++i) {
                        C[i+j*ldC] = 0.0;
                    }
                } else if (beta!=1.0) {
                    for (i=0; i<m; ++i) {
                        C[i+j*ldC] *= beta;
                    }
                }
                for (l=0; l<k; ++l) {
                    if (B[j+l*ldB]!=0.0) {
                        tmp = alpha*B[j+l*ldB];
                        for (i=0; i<m; ++i) {
                            C[i+j*ldC] += tmp*A[i+l*ldA];
                        }
                    }
                }
            }
#elif defined(ULM_BLOCKED)
            dgemm_nn(m, n, k,
                     alpha,
                     A, 1, ldA,
                     B, ldB, 1,
                     beta,
                     C, 1, ldC);
#else
#error      "no implementation specified!\n"
#endif
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
#if defined(ULM_REFERENCE)
            for (j=0; j<n; ++j) {
                for (i=0; i<m; ++i) {
                    tmp = 0.0;
                    for (l=0; l<k; ++l) {
                        tmp += A[l+i*ldA]*B[j+l*ldB];
                    }
                    if (beta==0.0) {
                        C[i+j*ldC] = alpha*tmp;
                    } else {
                        C[i+j*ldC] = alpha*tmp + beta*C[i+j*ldC];
                    }
                }
            }
#elif defined(ULM_BLOCKED)
            dgemm_nn(m, n, k,
                     alpha,
                     A, ldA, 1,
                     B, ldB, 1,
                     beta,
                     C, 1, ldC);
#else
#error      "no implementation specified!\n"
#endif
        }
    }
}

void
F77BLAS(dgemm)(const char     *_transA,
               const char     *_transB,
               const int      *_m,
               const int      *_n,
               const int      *_k,
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
    enum Trans transA = charToTranspose(*_transA);
    enum Trans transB = charToTranspose(*_transB);
    int m             = *_m;
    int n             = *_n;
    int k             = *_k;
    double alpha      = *_alpha;
    int ldA           = *_ldA;
    int ldB           = *_ldB;
    double beta       = *_beta;
    int ldC           = *_ldC;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (transA==NoTrans || transA==Conj) ? m : k;
    int numRowsB = (transB==NoTrans || transB==Conj) ? k : n;

//
//  Test the input parameters
//
    int info = 0;
    if (transA==0) {
        info = 1;
    } else if (transB==0) {
        info = 2;
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<max(1,numRowsA)) {
        info = 8;
    } else if (ldB<max(1,numRowsB)) {
        info = 10;
    } else if (ldC<max(1,m)) {
        info = 13;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DGEMM ", &info);
    }

    ULMBLAS(dgemm)(transA, transB,
                   m, n, k,
                   alpha,
                   A, ldA,
                   B, ldB,
                   beta,
                   C, ldC);
}
