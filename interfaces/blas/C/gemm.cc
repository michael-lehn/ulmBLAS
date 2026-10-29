#include BLAS_HEADER
#include <complex>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(sgemm)(enum CBLAS_TRANSPOSE  transA,
               enum CBLAS_TRANSPOSE  transB,
               ULMBLAS_INDEX_TYPE    m,
               ULMBLAS_INDEX_TYPE    n,
               ULMBLAS_INDEX_TYPE    k,
               float                 alpha,
               const float           *A,
               ULMBLAS_INDEX_TYPE    ldA,
               const float           *B,
               ULMBLAS_INDEX_TYPE    ldB,
               float                 beta,
               float                 *C,
               ULMBLAS_INDEX_TYPE    ldC)
{
//
//  Start the operations.
//
    if (transB==CblasNoTrans || transB==AtlasConj) {
        if (transA==CblasNoTrans || transA==AtlasConj) {
//
//          Form  C := alpha*A*B + beta*C.
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, 1, ldA,
                                          false, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, ldA, 1,
                                          false, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        }
    } else {
        if (transA==CblasNoTrans || transA==AtlasConj) {
//
//          Form  C := alpha*A*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, 1, ldA,
                                          false, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, ldA, 1,
                                          false, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        }
    }
}


void
ULMBLAS(dgemm)(enum CBLAS_TRANSPOSE  transA,
               enum CBLAS_TRANSPOSE  transB,
               ULMBLAS_INDEX_TYPE    m,
               ULMBLAS_INDEX_TYPE    n,
               ULMBLAS_INDEX_TYPE    k,
               double                alpha,
               const double          *A,
               ULMBLAS_INDEX_TYPE    ldA,
               const double          *B,
               ULMBLAS_INDEX_TYPE    ldB,
               double                beta,
               double                *C,
               ULMBLAS_INDEX_TYPE    ldC)
{
//
//  Start the operations.
//
    if (transB==CblasNoTrans || transB==AtlasConj) {
        if (transA==CblasNoTrans || transA==AtlasConj) {
//
//          Form  C := alpha*A*B + beta*C.
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, 1, ldA,
                                          false, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, ldA, 1,
                                          false, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        }
    } else {
        if (transA==CblasNoTrans || transA==AtlasConj) {
//
//          Form  C := alpha*A*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, 1, ldA,
                                          false, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, ldA, 1,
                                          false, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        }
    }
}

void
ULMBLAS(cgemm)(enum CBLAS_TRANSPOSE  transA,
               enum CBLAS_TRANSPOSE  transB,
               ULMBLAS_INDEX_TYPE    m,
               ULMBLAS_INDEX_TYPE    n,
               ULMBLAS_INDEX_TYPE    k,
               const float           *alpha_,
               const float           *A_,
               ULMBLAS_INDEX_TYPE    ldA,
               const float           *B_,
               ULMBLAS_INDEX_TYPE    ldB,
               const float           *beta_,
               float                 *C_,
               ULMBLAS_INDEX_TYPE    ldC)
{
    typedef std::complex<float> fcomplex;

    bool conjA   = (transA == AtlasConj || transA == CblasConjTrans);
    bool conjB   = (transB == AtlasConj || transB == CblasConjTrans);

    fcomplex    alpha(alpha_[0], alpha_[1]);
    fcomplex    beta(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *B = reinterpret_cast<const fcomplex *>(B_);
    fcomplex       *C = reinterpret_cast<fcomplex *>(C_);

//
//  Start the operations.
//
    if (transB==CblasNoTrans || transB==AtlasConj) {
        if (transA==CblasNoTrans || transA==AtlasConj) {
//
//          Form  C := alpha*A*B + beta*C.
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, 1, ldA,
                                          conjB, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, ldA, 1,
                                          conjB, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        }
    } else {
        if (transA==CblasNoTrans || transA==AtlasConj) {
//
//          Form  C := alpha*A*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, 1, ldA,
                                          conjB, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, ldA, 1,
                                          conjB, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        }
    }
}


void
ULMBLAS(zgemm)(enum CBLAS_TRANSPOSE  transA,
               enum CBLAS_TRANSPOSE  transB,
               ULMBLAS_INDEX_TYPE    m,
               ULMBLAS_INDEX_TYPE    n,
               ULMBLAS_INDEX_TYPE    k,
               const double          *alpha_,
               const double          *A_,
               ULMBLAS_INDEX_TYPE    ldA,
               const double          *B_,
               ULMBLAS_INDEX_TYPE    ldB,
               const double          *beta_,
               double                *C_,
               ULMBLAS_INDEX_TYPE    ldC)
{
    typedef std::complex<double> dcomplex;

    bool conjA   = (transA == AtlasConj || transA == CblasConjTrans);
    bool conjB   = (transB == AtlasConj || transB == CblasConjTrans);

    dcomplex    alpha(alpha_[0], alpha_[1]);
    dcomplex    beta(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *B = reinterpret_cast<const dcomplex *>(B_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);

//
//  Start the operations.
//
    if (transB==CblasNoTrans || transB==AtlasConj) {
        if (transA==CblasNoTrans || transA==AtlasConj) {
//
//          Form  C := alpha*A*B + beta*C.
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, 1, ldA,
                                          conjB, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, ldA, 1,
                                          conjB, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        }
    } else {
        if (transA==CblasNoTrans || transA==AtlasConj) {
//
//          Form  C := alpha*A*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, 1, ldA,
                                          conjB, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, ldA, 1,
                                          conjB, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        }
    }
}

void
CBLAS(sgemm)(enum CBLAS_ORDER      order,
             enum CBLAS_TRANSPOSE  transA,
             enum CBLAS_TRANSPOSE  transB,
             ULMBLAS_INDEX_TYPE    m,
             ULMBLAS_INDEX_TYPE    n,
             ULMBLAS_INDEX_TYPE    k,
             float                 alpha,
             const float           *A,
             ULMBLAS_INDEX_TYPE    ldA,
             const float           *B,
             ULMBLAS_INDEX_TYPE    ldB,
             float                 beta,
             float                 *C,
             ULMBLAS_INDEX_TYPE    ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA;
    ULMBLAS_INDEX_TYPE numRowsB;

    if (order==CblasColMajor) {
        numRowsA = (transA==CblasNoTrans || transA==AtlasConj) ? m : k;
        numRowsB = (transB==CblasNoTrans || transB==AtlasConj) ? k : n;
    } else {
        numRowsB = (transB==CblasNoTrans || transB==AtlasConj) ? n : k;
        numRowsA = (transA==CblasNoTrans || transA==AtlasConj) ? k : m;
    }


//
//  Test the input parameters
//
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (transA!=CblasNoTrans && transA!=CblasTrans
     && transA!=AtlasConj && transA!=CblasConjTrans)
    {
        info = 2;
    } else if (transB!=CblasNoTrans && transB!=CblasTrans
            && transB!=AtlasConj && transB!=CblasConjTrans)
    {
        info = 3;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 4;
            } else if (n<0) {
                info = 5;
            } else if (k<0) {
                info = 6;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 9;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
                info = 11;
            } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
                info = 14;
            }
        } else {
            if (n<0) {
                info = 4;
            } else if (m<0) {
                info = 5;
            } else if (k<0) {
                info = 6;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
                info = 9;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 11;
            } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
                info = 14;
            }
         }
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_sgemm", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(sgemm)(transA, transB, m, n, k, alpha,
                       A, ldA,
                       B, ldB,
                       beta,
                       C, ldC);
    } else {
        ULMBLAS(sgemm)(transB, transA, n, m, k, alpha,
                       B, ldB,
                       A, ldA,
                       beta,
                       C, ldC);
    }
}

void
CBLAS(dgemm)(enum CBLAS_ORDER      order,
             enum CBLAS_TRANSPOSE  transA,
             enum CBLAS_TRANSPOSE  transB,
             ULMBLAS_INDEX_TYPE    m,
             ULMBLAS_INDEX_TYPE    n,
             ULMBLAS_INDEX_TYPE    k,
             double                alpha,
             const double          *A,
             ULMBLAS_INDEX_TYPE    ldA,
             const double          *B,
             ULMBLAS_INDEX_TYPE    ldB,
             double                beta,
             double                *C,
             ULMBLAS_INDEX_TYPE    ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA;
    ULMBLAS_INDEX_TYPE numRowsB;

    if (order==CblasColMajor) {
        numRowsA = (transA==CblasNoTrans || transA==AtlasConj) ? m : k;
        numRowsB = (transB==CblasNoTrans || transB==AtlasConj) ? k : n;
    } else {
        numRowsB = (transB==CblasNoTrans || transB==AtlasConj) ? n : k;
        numRowsA = (transA==CblasNoTrans || transA==AtlasConj) ? k : m;
    }


//
//  Test the input parameters
//
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (transA!=CblasNoTrans && transA!=CblasTrans
     && transA!=AtlasConj && transA!=CblasConjTrans)
    {
        info = 2;
    } else if (transB!=CblasNoTrans && transB!=CblasTrans
            && transB!=AtlasConj && transB!=CblasConjTrans)
    {
        info = 3;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 4;
            } else if (n<0) {
                info = 5;
            } else if (k<0) {
                info = 6;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 9;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
                info = 11;
            } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
                info = 14;
            }
        } else {
            if (n<0) {
                info = 4;
            } else if (m<0) {
                info = 5;
            } else if (k<0) {
                info = 6;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
                info = 9;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 11;
            } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
                info = 14;
            }
         }
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_dgemm", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(dgemm)(transA, transB, m, n, k, alpha,
                       A, ldA,
                       B, ldB,
                       beta,
                       C, ldC);
    } else {
        ULMBLAS(dgemm)(transB, transA, n, m, k, alpha,
                       B, ldB,
                       A, ldA,
                       beta,
                       C, ldC);
    }
}

void
CBLAS(cgemm)(enum CBLAS_ORDER      order,
             enum CBLAS_TRANSPOSE  transA,
             enum CBLAS_TRANSPOSE  transB,
             ULMBLAS_INDEX_TYPE    m,
             ULMBLAS_INDEX_TYPE    n,
             ULMBLAS_INDEX_TYPE    k,
             const float           *alpha,
             const float           *A,
             ULMBLAS_INDEX_TYPE    ldA,
             const float           *B,
             ULMBLAS_INDEX_TYPE    ldB,
             const float           *beta,
             float                 *C,
             ULMBLAS_INDEX_TYPE    ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA;
    ULMBLAS_INDEX_TYPE numRowsB;

    if (order==CblasColMajor) {
        numRowsA = (transA==CblasNoTrans || transA==AtlasConj) ? m : k;
        numRowsB = (transB==CblasNoTrans || transB==AtlasConj) ? k : n;
    } else {
        numRowsB = (transB==CblasNoTrans || transB==AtlasConj) ? n : k;
        numRowsA = (transA==CblasNoTrans || transA==AtlasConj) ? k : m;
    }


//
//  Test the input parameters
//
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (transA!=CblasNoTrans && transA!=CblasTrans
     && transA!=AtlasConj && transA!=CblasConjTrans)
    {
        info = 2;
    } else if (transB!=CblasNoTrans && transB!=CblasTrans
            && transB!=AtlasConj && transB!=CblasConjTrans)
    {
        info = 3;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 4;
            } else if (n<0) {
                info = 5;
            } else if (k<0) {
                info = 6;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 9;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
                info = 11;
            } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
                info = 14;
            }
        } else {
            if (n<0) {
                info = 4;
            } else if (m<0) {
                info = 5;
            } else if (k<0) {
                info = 6;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
                info = 9;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 11;
            } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
                info = 14;
            }
         }
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_cgemm", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(cgemm)(transA, transB, m, n, k, alpha,
                       A, ldA,
                       B, ldB,
                       beta,
                       C, ldC);
    } else {
        ULMBLAS(cgemm)(transB, transA, n, m, k, alpha,
                       B, ldB,
                       A, ldA,
                       beta,
                       C, ldC);
    }
}

void
CBLAS(zgemm)(enum CBLAS_ORDER      order,
             enum CBLAS_TRANSPOSE  transA,
             enum CBLAS_TRANSPOSE  transB,
             ULMBLAS_INDEX_TYPE    m,
             ULMBLAS_INDEX_TYPE    n,
             ULMBLAS_INDEX_TYPE    k,
             const double          *alpha,
             const double          *A,
             ULMBLAS_INDEX_TYPE    ldA,
             const double          *B,
             ULMBLAS_INDEX_TYPE    ldB,
             const double          *beta,
             double                *C,
             ULMBLAS_INDEX_TYPE    ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA;
    ULMBLAS_INDEX_TYPE numRowsB;

    if (order==CblasColMajor) {
        numRowsA = (transA==CblasNoTrans || transA==AtlasConj) ? m : k;
        numRowsB = (transB==CblasNoTrans || transB==AtlasConj) ? k : n;
    } else {
        numRowsB = (transB==CblasNoTrans || transB==AtlasConj) ? n : k;
        numRowsA = (transA==CblasNoTrans || transA==AtlasConj) ? k : m;
    }


//
//  Test the input parameters
//
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (transA!=CblasNoTrans && transA!=CblasTrans
     && transA!=AtlasConj && transA!=CblasConjTrans)
    {
        info = 2;
    } else if (transB!=CblasNoTrans && transB!=CblasTrans
            && transB!=AtlasConj && transB!=CblasConjTrans)
    {
        info = 3;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 4;
            } else if (n<0) {
                info = 5;
            } else if (k<0) {
                info = 6;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 9;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
                info = 11;
            } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
                info = 14;
            }
        } else {
            if (n<0) {
                info = 4;
            } else if (m<0) {
                info = 5;
            } else if (k<0) {
                info = 6;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
                info = 9;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 11;
            } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
                info = 14;
            }
         }
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_zgemm", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(zgemm)(transA, transB, m, n, k, alpha,
                       A, ldA,
                       B, ldB,
                       beta,
                       C, ldC);
    } else {
        ULMBLAS(zgemm)(transB, transA, n, m, k, alpha,
                       B, ldB,
                       A, ldA,
                       beta,
                       C, ldC);
    }
}

} // extern "C"
