#include BLAS_HEADER
#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(ssyrk)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               int                   n,
               int                   k,
               float                 alpha,
               const float           *A,
               int                   ldA,
               float                 beta,
               float                 *C,
               int                   ldC)
{
//
//  Start the operations.
//
    if (trans==CblasNoTrans || trans==AtlasConj) {
        if (upLo==CblasLower) {
            ulmBLAS::sylrk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        } else {
            ulmBLAS::syurk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::syurk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::sylrk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        }
    }
}

void
ULMBLAS(dsyrk)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               int                   n,
               int                   k,
               double                alpha,
               const double          *A,
               int                   ldA,
               double                beta,
               double                *C,
               int                   ldC)
{
//
//  Start the operations.
//
    if (trans==CblasNoTrans || trans==AtlasConj) {
        if (upLo==CblasLower) {
            ulmBLAS::sylrk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        } else {
            ulmBLAS::syurk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::syurk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::sylrk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        }
    }
}

void
ULMBLAS(csyrk)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               int                   n,
               int                   k,
               const float           *alpha_,
               const float           *A_,
               int                   ldA,
               const float           *beta_,
               float                 *C_,
               int                   ldC)
{
    typedef std::complex<float> fcomplex;

    fcomplex   alpha(alpha_[0], alpha_[1]);
    fcomplex   beta(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    fcomplex       *C = reinterpret_cast<fcomplex *>(C_);
//
//  Start the operations.
//
    if (trans==CblasNoTrans) {
        if (upLo==CblasLower) {
            ulmBLAS::sylrk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        } else {
            ulmBLAS::syurk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::syurk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::sylrk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        }
    }
}

void
ULMBLAS(zsyrk)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               int                   n,
               int                   k,
               const double          *alpha_,
               const double          *A_,
               int                   ldA,
               const double          *beta_,
               double                *C_,
               int                   ldC)
{
    typedef std::complex<double> dcomplex;

    dcomplex   alpha(alpha_[0], alpha_[1]);
    dcomplex   beta(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);
//
//  Start the operations.
//
    if (trans==CblasNoTrans) {
        if (upLo==CblasLower) {
            ulmBLAS::sylrk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        } else {
            ulmBLAS::syurk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::syurk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::sylrk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        }
    }
}

void
CBLAS(ssyrk)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             int                   n,
             int                   k,
             float                 alpha,
             const float           *A,
             int                   ldA,
             float                 beta,
             float                 *C,
             int                   ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA;

    if (order==CblasColMajor) {
        numRowsA = (trans==CblasNoTrans || trans==AtlasConj) ? n : k;
    } else {
        numRowsA = (trans==CblasNoTrans || trans==AtlasConj) ? k : n;
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
    } else if (ldC<std::max(1,n)) {
        info = 11;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_ssyrk", "");
    }

    if (order==CblasColMajor) {
        ULMBLAS(ssyrk)(upLo, trans, n, k, alpha, A, ldA, beta, C, ldC);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(ssyrk)(upLo, trans, n, k, alpha, A, ldA, beta, C, ldC);
    }
}

void
CBLAS(dsyrk)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             int                   n,
             int                   k,
             double                alpha,
             const double          *A,
             int                   ldA,
             double                beta,
             double                *C,
             int                   ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA;

    if (order==CblasColMajor) {
        numRowsA = (trans==CblasNoTrans || trans==AtlasConj) ? n : k;
    } else {
        numRowsA = (trans==CblasNoTrans || trans==AtlasConj) ? k : n;
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
    } else if (ldC<std::max(1,n)) {
        info = 11;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_dsyrk", "");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dsyrk)(upLo, trans, n, k, alpha, A, ldA, beta, C, ldC);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(dsyrk)(upLo, trans, n, k, alpha, A, ldA, beta, C, ldC);
    }
}

void
CBLAS(csyrk)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             int                   n,
             int                   k,
             const float           *alpha,
             const float           *A,
             int                   ldA,
             const float           *beta,
             float                 *C,
             int                   ldC)
{
    typedef std::complex<float> fcomplex;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA;

    if (order==CblasColMajor) {
        numRowsA = (trans==CblasNoTrans) ? n : k;
    } else {
        numRowsA = (trans==CblasNoTrans) ? k : n;
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
    } else if (ldC<std::max(1,n)) {
        info = 11;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_csyrk", "");
    }

    if (order==CblasColMajor) {
        ULMBLAS(csyrk)(upLo, trans, n, k, alpha, A, ldA, beta, C, ldC);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(csyrk)(upLo, trans, n, k, alpha, A, ldA, beta, C, ldC);
    }
}


void
CBLAS(zsyrk)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             int                   n,
             int                   k,
             const double          *alpha,
             const double          *A,
             int                   ldA,
             const double          *beta,
             double                *C,
             int                   ldC)
{
    typedef std::complex<double> dcomplex;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA;

    if (order==CblasColMajor) {
        numRowsA = (trans==CblasNoTrans) ? n : k;
    } else {
        numRowsA = (trans==CblasNoTrans) ? k : n;
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
    } else if (ldC<std::max(1,n)) {
        info = 11;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_zsyrk", "");
    }

    if (order==CblasColMajor) {
        ULMBLAS(zsyrk)(upLo, trans, n, k, alpha, A, ldA, beta, C, ldC);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(zsyrk)(upLo, trans, n, k, alpha, A, ldA, beta, C, ldC);
    }
}


} // extern "C"
