#include BLAS_HEADER
#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(chemm)(enum CBLAS_SIDE  side,
               enum CBLAS_UPLO  upLo,
               int              m,
               int              n,
               float            *alpha_,
               const float      *A_,
               int              ldA,
               float            *B_,
               int              ldB,
               float            *beta_,
               float            *C_,
               int              ldC)
{
    typedef std::complex<float> fcomplex;

    bool left     = (side==CblasLeft);
    bool lower    = (upLo==CblasLower);

    fcomplex   alpha(alpha_[0], alpha_[1]);
    fcomplex   beta(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *B = reinterpret_cast<const fcomplex *>(B_);
    fcomplex       *C = reinterpret_cast<fcomplex *>(C_);
//
//  Start the operations.
//
    if (left) {
        if (lower) {
            ulmBLAS::helmm(m, n, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        } else {
            ulmBLAS::heumm(m, n, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::heumm(n, m, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::helmm(n, m, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        }
    }
}

void
ULMBLAS(zhemm)(enum CBLAS_SIDE  side,
               enum CBLAS_UPLO  upLo,
               int              m,
               int              n,
               double           *alpha_,
               const double     *A_,
               int              ldA,
               double           *B_,
               int              ldB,
               double           *beta_,
               double           *C_,
               int              ldC)
{
    typedef std::complex<double> dcomplex;

    bool left     = (side==CblasLeft);
    bool lower    = (upLo==CblasLower);

    dcomplex   alpha(alpha_[0], alpha_[1]);
    dcomplex   beta(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *B = reinterpret_cast<const dcomplex *>(B_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);
//
//  Start the operations.
//
    if (left) {
        if (lower) {
            ulmBLAS::helmm(m, n, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        } else {
            ulmBLAS::heumm(m, n, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::heumm(n, m, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::helmm(n, m, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        }
    }
}

void
CBLAS(chemm)(enum CBLAS_ORDER  order,
             enum CBLAS_SIDE   side,
             enum CBLAS_UPLO   upLo,
             int               m,
             int               n,
             float             *alpha,
             const float       *A,
             int               ldA,
             float             *B,
             int               ldB,
             float             *beta,
             float             *C,
             int               ldC)
{
    bool left = (side==CblasLeft);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int  numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (side!=CblasLeft && side!=CblasRight) {
        info = 2;
    } else if (upLo!=CblasLower && upLo!=CblasUpper) {
        info = 3;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 4;
            } else if (n<0) {
                info = 5;
            } else if (ldA<std::max(1,numRowsA)) {
                info = 8;
            } else if (ldB<std::max(1,m)) {
                info = 10;
            } else if (ldC<std::max(1,m)) {
                info = 13;
            }
        } else {
            if (n<0) {
                info = 4;
            } else if (m<0) {
                info = 5;
            } else if (ldA<std::max(1,numRowsA)) {
                info = 8;
            } else if (ldB<std::max(1,n)) {
                info = 10;
            } else if (ldC<std::max(1,n)) {
                info = 13;
            }
        }
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_chemm", "");
    }

    if (order==CblasColMajor) {
        ULMBLAS(chemm)(side, upLo, m, n, alpha, A, ldA, B, ldB, beta, C, ldC);
    } else {
        side = (side==CblasLeft) ? CblasRight : CblasLeft;
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(chemm)(side, upLo, n, m, alpha, A, ldA, B, ldB, beta, C, ldC);
    }
}

void
CBLAS(zhemm)(enum CBLAS_ORDER  order,
             enum CBLAS_SIDE   side,
             enum CBLAS_UPLO   upLo,
             int               m,
             int               n,
             double            *alpha,
             const double      *A,
             int               ldA,
             double            *B,
             int               ldB,
             double            *beta,
             double            *C,
             int               ldC)
{
    bool left = (side==CblasLeft);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int  numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (side!=CblasLeft && side!=CblasRight) {
        info = 2;
    } else if (upLo!=CblasLower && upLo!=CblasUpper) {
        info = 3;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 4;
            } else if (n<0) {
                info = 5;
            } else if (ldA<std::max(1,numRowsA)) {
                info = 8;
            } else if (ldB<std::max(1,m)) {
                info = 10;
            } else if (ldC<std::max(1,m)) {
                info = 13;
            }
        } else {
            if (n<0) {
                info = 4;
            } else if (m<0) {
                info = 5;
            } else if (ldA<std::max(1,numRowsA)) {
                info = 8;
            } else if (ldB<std::max(1,n)) {
                info = 10;
            } else if (ldC<std::max(1,n)) {
                info = 13;
            }
        }
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_zhemm", "");
    }

    if (order==CblasColMajor) {
        ULMBLAS(zhemm)(side, upLo, m, n, alpha, A, ldA, B, ldB, beta, C, ldC);
    } else {
        side = (side==CblasLeft) ? CblasRight : CblasLeft;
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(zhemm)(side, upLo, n, m, alpha, A, ldA, B, ldB, beta, C, ldC);
    }
}

} // extern "C"
