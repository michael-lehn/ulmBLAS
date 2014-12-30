#include BLAS_HEADER
#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level3/sylrk.h>
#include <ulmblas/level3/syurk.h>

extern "C" {

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

} // extern "C"
