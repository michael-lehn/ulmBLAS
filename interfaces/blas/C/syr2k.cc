#include BLAS_HEADER
#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(ssyr2k)(enum CBLAS_UPLO       upLo,
                enum CBLAS_TRANSPOSE  trans,
                int                   n,
                int                   k,
                float                 alpha,
                const float           *A,
                int                   ldA,
                const float           *B,
                int                   ldB,
                float                 beta,
                float                 *C,
                int                   ldC)
{
//
//  Start the operations.
//
    if (trans==CblasNoTrans || trans==AtlasConj) {
        if (upLo==CblasLower) {
            ulmBLAS::sylr2k(n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        } else {
            ulmBLAS::syur2k(n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::syur2k(n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::sylr2k(n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        }
    }
}

void
ULMBLAS(dsyr2k)(enum CBLAS_UPLO       upLo,
                enum CBLAS_TRANSPOSE  trans,
                int                   n,
                int                   k,
                double                alpha,
                const double          *A,
                int                   ldA,
                const double          *B,
                int                   ldB,
                double                beta,
                double                *C,
                int                   ldC)
{
//
//  Start the operations.
//
    if (trans==CblasNoTrans || trans==AtlasConj) {
        if (upLo==CblasLower) {
            ulmBLAS::sylr2k(n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        } else {
            ulmBLAS::syur2k(n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::syur2k(n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::sylr2k(n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        }
    }
}

void
ULMBLAS(csyr2k)(enum CBLAS_UPLO       upLo,
                enum CBLAS_TRANSPOSE  trans,
                int                   n,
                int                   k,
                const float           *alpha_,
                const float           *A_,
                int                   ldA,
                const float           *B_,
                int                   ldB,
                const float           *beta_,
                float                 *C_,
                int                   ldC)
{
    typedef std::complex<float> fcomplex;

    fcomplex   alpha(alpha_[0], alpha_[1]);
    fcomplex   beta(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *B = reinterpret_cast<const fcomplex *>(B_);
    fcomplex       *C = reinterpret_cast<fcomplex *>(C_);

//
//  Start the operations.
//
    if (trans==CblasNoTrans || trans==AtlasConj) {
        if (upLo==CblasLower) {
            ulmBLAS::sylr2k(n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        } else {
            ulmBLAS::syur2k(n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::syur2k(n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::sylr2k(n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        }
    }
}

void
ULMBLAS(zsyr2k)(enum CBLAS_UPLO       upLo,
                enum CBLAS_TRANSPOSE  trans,
                int                   n,
                int                   k,
                const double          *alpha_,
                const double          *A_,
                int                   ldA,
                const double          *B_,
                int                   ldB,
                const double          *beta_,
                double                *C_,
                int                   ldC)
{
    typedef std::complex<double> dcomplex;

    dcomplex   alpha(alpha_[0], alpha_[1]);
    dcomplex   beta(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *B = reinterpret_cast<const dcomplex *>(B_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);

//
//  Start the operations.
//
    if (trans==CblasNoTrans || trans==AtlasConj) {
        if (upLo==CblasLower) {
            ulmBLAS::sylr2k(n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        } else {
            ulmBLAS::syur2k(n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::syur2k(n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::sylr2k(n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        }
    }
}

void
CBLAS(ssyr2k)(enum CBLAS_ORDER      order,
              enum CBLAS_UPLO       upLo,
              enum CBLAS_TRANSPOSE  trans,
              int                   n,
              int                   k,
              float                 alpha,
              const float           *A,
              int                   ldA,
              const float           *B,
              int                   ldB,
              float                 beta,
              float                 *C,
              int                   ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA;
    int numRowsB;

    if (order==CblasColMajor) {
        numRowsA = (trans==CblasNoTrans || trans==AtlasConj) ? n : k;
        numRowsB = (trans==CblasNoTrans || trans==AtlasConj) ? n : k;
    } else {
        numRowsA = (trans==CblasNoTrans || trans==AtlasConj) ? k : n;
        numRowsB = (trans==CblasNoTrans || trans==AtlasConj) ? k : n;
    }

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (upLo!=CblasLower && upLo!=CblasUpper) {
        info = 2;
    } else if (trans!=CblasNoTrans && trans!=AtlasConj
            && trans!=CblasTrans && trans!=CblasConjTrans)
    {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max(1,n)) {
        info = 13;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_ssyr2k", "");
    }

    if (order==CblasColMajor) {
        ULMBLAS(ssyr2k)(upLo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(ssyr2k)(upLo, trans, n, k, alpha, B, ldB, A, ldA, beta, C, ldC);
    }
}

void
CBLAS(dsyr2k)(enum CBLAS_ORDER      order,
              enum CBLAS_UPLO       upLo,
              enum CBLAS_TRANSPOSE  trans,
              int                   n,
              int                   k,
              double                alpha,
              const double          *A,
              int                   ldA,
              const double          *B,
              int                   ldB,
              double                beta,
              double                *C,
              int                   ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA;
    int numRowsB;

    if (order==CblasColMajor) {
        numRowsA = (trans==CblasNoTrans || trans==AtlasConj) ? n : k;
        numRowsB = (trans==CblasNoTrans || trans==AtlasConj) ? n : k;
    } else {
        numRowsA = (trans==CblasNoTrans || trans==AtlasConj) ? k : n;
        numRowsB = (trans==CblasNoTrans || trans==AtlasConj) ? k : n;
    }

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (upLo!=CblasLower && upLo!=CblasUpper) {
        info = 2;
    } else if (trans!=CblasNoTrans && trans!=AtlasConj
            && trans!=CblasTrans && trans!=CblasConjTrans)
    {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max(1,n)) {
        info = 13;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_dsyr2k", "");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dsyr2k)(upLo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(dsyr2k)(upLo, trans, n, k, alpha, B, ldB, A, ldA, beta, C, ldC);
    }
}

void
CBLAS(csyr2k)(enum CBLAS_ORDER      order,
              enum CBLAS_UPLO       upLo,
              enum CBLAS_TRANSPOSE  trans,
              int                   n,
              int                   k,
              const float           *alpha,
              const float           *A,
              int                   ldA,
              const float           *B,
              int                   ldB,
              const float           *beta,
              float                 *C,
              int                   ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA;
    int numRowsB;

    if (order==CblasColMajor) {
        numRowsA = (trans==CblasNoTrans || trans==AtlasConj) ? n : k;
        numRowsB = (trans==CblasNoTrans || trans==AtlasConj) ? n : k;
    } else {
        numRowsA = (trans==CblasNoTrans || trans==AtlasConj) ? k : n;
        numRowsB = (trans==CblasNoTrans || trans==AtlasConj) ? k : n;
    }

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (upLo!=CblasLower && upLo!=CblasUpper) {
        info = 2;
    } else if (trans!=CblasNoTrans && trans!=CblasTrans) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max(1,n)) {
        info = 13;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_csyr2k", "");
    }

    if (order==CblasColMajor) {
        ULMBLAS(csyr2k)(upLo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(csyr2k)(upLo, trans, n, k, alpha, B, ldB, A, ldA, beta, C, ldC);
    }
}

void
CBLAS(zsyr2k)(enum CBLAS_ORDER      order,
              enum CBLAS_UPLO       upLo,
              enum CBLAS_TRANSPOSE  trans,
              int                   n,
              int                   k,
              const double          *alpha,
              const double          *A,
              int                   ldA,
              const double          *B,
              int                   ldB,
              const double          *beta,
              double                *C,
              int                   ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA;
    int numRowsB;

    if (order==CblasColMajor) {
        numRowsA = (trans==CblasNoTrans || trans==AtlasConj) ? n : k;
        numRowsB = (trans==CblasNoTrans || trans==AtlasConj) ? n : k;
    } else {
        numRowsA = (trans==CblasNoTrans || trans==AtlasConj) ? k : n;
        numRowsB = (trans==CblasNoTrans || trans==AtlasConj) ? k : n;
    }

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (upLo!=CblasLower && upLo!=CblasUpper) {
        info = 2;
    } else if (trans!=CblasNoTrans && trans!=CblasTrans) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max(1,n)) {
        info = 13;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_zsyr2k", "");
    }

    if (order==CblasColMajor) {
        ULMBLAS(zsyr2k)(upLo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(zsyr2k)(upLo, trans, n, k, alpha, B, ldB, A, ldA, beta, C, ldC);
    }
}

} // extern "C"
