#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level2/tplmv.h>
#include <ulmblas/level2/tplmtv.h>
#include <ulmblas/level2/tpumv.h>
#include <ulmblas/level2/tpumtv.h>

#include <iostream>

extern "C" {

void
ULMBLAS(dtpmv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               enum CBLAS_DIAG       diag,
               int                   n,
               const double          *A,
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
            ulmBLAS::tplmv(n, unitDiag, A, x, incX);
        } else {
            ulmBLAS::tpumv(n, unitDiag, A, x, incX);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::tplmtv(n, unitDiag, A, x, incX);
        } else {
            ulmBLAS::tpumtv(n, unitDiag, A, x, incX);
        }
    }
}

void
CBLAS(dtpmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             int                   n,
             const double          *A,
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
    } else if (incX==0) {
        info = 8;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dtpmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dtpmv)(upLo, trans, diag, n, A, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(dtpmv)(upLo, trans, diag, n, A, x, incX);
    }
}

} // extern "C"
