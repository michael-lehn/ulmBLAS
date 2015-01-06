#include BLAS_HEADER
#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level3/helrk.h>
#include <ulmblas/level3/heurk.h>

extern "C" {

void
ULMBLAS(zherk)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               int                   n,
               int                   k,
               double                alpha,
               const double          *A_,
               int                   ldA,
               double                beta,
               double                *C_,
               int                   ldC)
{
    typedef std::complex<double> dcomplex;

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);
//
//  Start the operations.
//
    if (trans==CblasNoTrans) {
        if (upLo==CblasLower) {
            ulmBLAS::helrk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        } else {
            ulmBLAS::heurk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::heurk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::helrk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        }
    }
}


void
CBLAS(zherk)(enum CBLAS_ORDER      order,
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
    } else if (trans!=CblasNoTrans && trans!=CblasConjTrans) {
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
        CBLAS(xerbla)(info, "cblas_zherk", "");
    }

    if (order==CblasColMajor) {
        ULMBLAS(zherk)(upLo, trans, n, k, alpha, A, ldA, beta, C, ldC);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = (trans==CblasConjTrans) ? CblasNoTrans : CblasConjTrans;
        ULMBLAS(zherk)(upLo, trans, n, k, alpha, A, ldA, beta, C, ldC);
    }
}


} // extern "C"
