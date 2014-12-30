#include BLAS_HEADER
#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level3/sylr2k.h>
#include <ulmblas/level3/syur2k.h>

extern "C" {

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

} // extern "C"
