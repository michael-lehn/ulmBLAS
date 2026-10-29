#include BLAS_HEADER
#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(cher2k)(enum CBLAS_UPLO       upLo,
                enum CBLAS_TRANSPOSE  trans,
                ULMBLAS_INDEX_TYPE    n,
                ULMBLAS_INDEX_TYPE    k,
                float                 *alpha_,
                const float           *A_,
                ULMBLAS_INDEX_TYPE    ldA,
                const float           *B_,
                ULMBLAS_INDEX_TYPE    ldB,
                float                 beta,
                float                 *C_,
                ULMBLAS_INDEX_TYPE    ldC)
{
    typedef std::complex<float> fcomplex;

    fcomplex   alpha(alpha_[0], alpha_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *B = reinterpret_cast<const fcomplex *>(B_);
    fcomplex       *C = reinterpret_cast<fcomplex *>(C_);

//
//  Start the operations.
//
    if (trans==CblasNoTrans) {
        if (upLo==CblasLower) {
            ulmBLAS::helr2k<std::ptrdiff_t>(n, k, alpha, A, 1, ldA, B, 1, ldB,
                                            beta, C, 1, ldC);
        } else {
            ulmBLAS::heur2k<std::ptrdiff_t>(n, k, alpha, A, 1, ldA, B, 1, ldB,
                                            beta, C, 1, ldC);
        }
    } else {
        alpha = ulmBLAS::conjugate(alpha);
        if (upLo==CblasLower) {
            ulmBLAS::heur2k<std::ptrdiff_t>(n, k, alpha, A, ldA, 1, B, ldB, 1,
                                            beta, C, ldC, 1);
        } else {
            ulmBLAS::helr2k<std::ptrdiff_t>(n, k, alpha, A, ldA, 1, B, ldB, 1,
                                            beta, C, ldC, 1);
        }
    }
}

void
ULMBLAS(zher2k)(enum CBLAS_UPLO       upLo,
                enum CBLAS_TRANSPOSE  trans,
                ULMBLAS_INDEX_TYPE    n,
                ULMBLAS_INDEX_TYPE    k,
                double                *alpha_,
                const double          *A_,
                ULMBLAS_INDEX_TYPE    ldA,
                const double          *B_,
                ULMBLAS_INDEX_TYPE    ldB,
                double                beta,
                double                *C_,
                ULMBLAS_INDEX_TYPE    ldC)
{
    typedef std::complex<double> dcomplex;

    dcomplex   alpha(alpha_[0], alpha_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *B = reinterpret_cast<const dcomplex *>(B_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);

//
//  Start the operations.
//
    if (trans==CblasNoTrans) {
        if (upLo==CblasLower) {
            ulmBLAS::helr2k<std::ptrdiff_t>(n, k, alpha, A, 1, ldA, B, 1, ldB,
                                            beta, C, 1, ldC);
        } else {
            ulmBLAS::heur2k<std::ptrdiff_t>(n, k, alpha, A, 1, ldA, B, 1, ldB,
                                            beta, C, 1, ldC);
        }
    } else {
        alpha = ulmBLAS::conjugate(alpha);
        if (upLo==CblasLower) {
            ulmBLAS::heur2k<std::ptrdiff_t>(n, k, alpha, A, ldA, 1, B, ldB, 1,
                                            beta, C, ldC, 1);
        } else {
            ulmBLAS::helr2k<std::ptrdiff_t>(n, k, alpha, A, ldA, 1, B, ldB, 1,
                                            beta, C, ldC, 1);
        }
    }
}


void
CBLAS(cher2k)(enum CBLAS_ORDER      order,
              enum CBLAS_UPLO       upLo,
              enum CBLAS_TRANSPOSE  trans,
              ULMBLAS_INDEX_TYPE    n,
              ULMBLAS_INDEX_TYPE    k,
              float                 *alpha,
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
        numRowsA = (trans==CblasNoTrans) ? n : k;
        numRowsB = (trans==CblasNoTrans) ? n : k;
    } else {
        numRowsA = (trans==CblasNoTrans) ? k : n;
        numRowsB = (trans==CblasNoTrans) ? k : n;
    }

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (upLo!=CblasLower && upLo!=CblasUpper) {
        info = 2;
    } else if (trans!=CblasNoTrans && trans!=CblasConjTrans) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 13;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_cher2k", "");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(cher2k)(upLo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = (trans==CblasConjTrans) ? CblasNoTrans : CblasConjTrans;
        alpha[1] = -alpha[1];
        ULMBLAS(cher2k)(upLo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
        alpha[1] = -alpha[1];
    }
}

void
CBLAS(zher2k)(enum CBLAS_ORDER      order,
              enum CBLAS_UPLO       upLo,
              enum CBLAS_TRANSPOSE  trans,
              ULMBLAS_INDEX_TYPE    n,
              ULMBLAS_INDEX_TYPE    k,
              double                *alpha,
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
        numRowsA = (trans==CblasNoTrans) ? n : k;
        numRowsB = (trans==CblasNoTrans) ? n : k;
    } else {
        numRowsA = (trans==CblasNoTrans) ? k : n;
        numRowsB = (trans==CblasNoTrans) ? k : n;
    }

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (upLo!=CblasLower && upLo!=CblasUpper) {
        info = 2;
    } else if (trans!=CblasNoTrans && trans!=CblasConjTrans) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 13;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_zher2k", "");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(zher2k)(upLo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = (trans==CblasConjTrans) ? CblasNoTrans : CblasConjTrans;
        alpha[1] = -alpha[1];
        ULMBLAS(zher2k)(upLo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
        alpha[1] = -alpha[1];
    }
}

} // extern "C"
