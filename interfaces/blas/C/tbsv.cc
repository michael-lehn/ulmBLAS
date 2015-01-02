#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level2/tblsv.h>
#include <ulmblas/level2/tblstv.h>
#include <ulmblas/level2/tbusv.h>
#include <ulmblas/level2/tbustv.h>

#include <iostream>

extern "C" {

void
ULMBLAS(dtbsv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               enum CBLAS_DIAG       diag,
               int                   n,
               int                   k,
               const double          *A,
               int                   ldA,
               double                *x,
               int                   incX)
{
    if (incX<0) {
        x -= incX*(n-1);
    }

    bool unitDiag = (diag==CblasUnit);

//
//  Start the operations.
//
    if (trans==CblasNoTrans || trans==AtlasConj) {
        if (upLo==CblasLower) {
            ulmBLAS::tblsv(n, k, unitDiag, A, ldA, x, incX);
        } else {
            ulmBLAS::tbusv(n, k, unitDiag, A, ldA, x, incX);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::tblstv(n, k, unitDiag, A, ldA, x, incX);
        } else {
            ulmBLAS::tbustv(n, k, unitDiag, A, ldA, x, incX);
        }
    }
}

void
CBLAS(dtbsv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             int                   n,
             int                   k,
             const double          *A,
             int                   ldA,
             double                *x,
             int                   incX)
{
//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (upLo!=CblasUpper && upLo!=CblasLower) {
        info = 2;
    } else if (trans!=CblasNoTrans && trans!=CblasTrans
            && trans!=CblasConjTrans && trans!=AtlasConj)
    {
        info = 3;
    } else if (diag!=CblasNonUnit && diag!=CblasUnit) {
        info = 4;
     } else if (n<0) {
        info = 5;
    } else if (k<0) {
        info = 6;
    } else if (ldA<k+1) {
        info = 8;
    } else if (incX==0) {
        info = 10;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dtbsv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dtbsv)(upLo, trans, diag, n, k, A, ldA, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(dtbsv)(upLo, trans, diag, n, k, A, ldA, x, incX);
    }
}

} // extern "C"
