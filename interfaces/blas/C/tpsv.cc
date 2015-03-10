#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

#include <iostream>

extern "C" {

void
ULMBLAS(stpsv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               enum CBLAS_DIAG       diag,
               int                   n,
               const float           *A,
               float                 *x,
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
            ulmBLAS::tplsv(n, unitDiag, A, x, incX);
        } else {
            ulmBLAS::tpusv(n, unitDiag, A, x, incX);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::tplstv(n, unitDiag, A, x, incX);
        } else {
            ulmBLAS::tpustv(n, unitDiag, A, x, incX);
        }
    }
}

void
ULMBLAS(dtpsv)(enum CBLAS_UPLO       upLo,
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
            ulmBLAS::tplsv(n, unitDiag, A, x, incX);
        } else {
            ulmBLAS::tpusv(n, unitDiag, A, x, incX);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::tplstv(n, unitDiag, A, x, incX);
        } else {
            ulmBLAS::tpustv(n, unitDiag, A, x, incX);
        }
    }
}

void
ULMBLAS(ctpsv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  transA_,
               enum CBLAS_DIAG       diag,
               int                   n,
               const float           *AP_,
               float                 *x_,
               int                   incX)
{
    bool lowerA = (upLo==CblasLower);
    bool transA = (transA_==CblasTrans || transA_==CblasConjTrans);
    bool conjA  = (transA_==AtlasConj || transA_==CblasConjTrans);

    typedef std::complex<float> fcomplex;
    const fcomplex *AP = reinterpret_cast<const fcomplex *>(AP_);
    fcomplex       *x  = reinterpret_cast<fcomplex *>(x_);

    if (incX<0) {
        x -= incX*(n-1);
    }

    bool unitDiag = (diag==CblasUnit);

//
//  Start the operations.
//
    if (!transA) {
        if (lowerA) {
            ulmBLAS::tplsv(n, unitDiag, conjA, AP, x, incX);
        } else {
            ulmBLAS::tpusv(n, unitDiag, conjA, AP, x, incX);
        }
    } else {
        if (lowerA) {
            ulmBLAS::tplstv(n, unitDiag, conjA, AP, x, incX);
        } else {
            ulmBLAS::tpustv(n, unitDiag, conjA, AP, x, incX);
        }
    }
}

void
ULMBLAS(ztpsv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  transA_,
               enum CBLAS_DIAG       diag,
               int                   n,
               const double          *AP_,
               double                *x_,
               int                   incX)
{
    bool lowerA = (upLo==CblasLower);
    bool transA = (transA_==CblasTrans || transA_==CblasConjTrans);
    bool conjA  = (transA_==AtlasConj || transA_==CblasConjTrans);

    typedef std::complex<double> dcomplex;
    const dcomplex *AP = reinterpret_cast<const dcomplex *>(AP_);
    dcomplex       *x  = reinterpret_cast<dcomplex *>(x_);

    if (incX<0) {
        x -= incX*(n-1);
    }

    bool unitDiag = (diag==CblasUnit);

//
//  Start the operations.
//
    if (!transA) {
        if (lowerA) {
            ulmBLAS::tplsv(n, unitDiag, conjA, AP, x, incX);
        } else {
            ulmBLAS::tpusv(n, unitDiag, conjA, AP, x, incX);
        }
    } else {
        if (lowerA) {
            ulmBLAS::tplstv(n, unitDiag, conjA, AP, x, incX);
        } else {
            ulmBLAS::tpustv(n, unitDiag, conjA, AP, x, incX);
        }
    }
}

void
CBLAS(stpsv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             int                   n,
             const float           *A,
             float                 *x,
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
        CBLAS(xerbla)(info, "cblas_stpsv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(stpsv)(upLo, trans, diag, n, A, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(stpsv)(upLo, trans, diag, n, A, x, incX);
    }
}

void
CBLAS(dtpsv)(enum CBLAS_ORDER      order,
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
        CBLAS(xerbla)(info, "cblas_dtpsv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dtpsv)(upLo, trans, diag, n, A, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(dtpsv)(upLo, trans, diag, n, A, x, incX);
    }
}

void
CBLAS(ctpsv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             int                   n,
             const float           *A,
             float                 *x,
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
        CBLAS(xerbla)(info, "cblas_ctpsv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(ctpsv)(upLo, trans, diag, n, A, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(ctpsv)(upLo, trans, diag, n, A, x, incX);
    }
}

void
CBLAS(ztpsv)(enum CBLAS_ORDER      order,
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
        CBLAS(xerbla)(info, "cblas_ztpsv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(ztpsv)(upLo, trans, diag, n, A, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(ztpsv)(upLo, trans, diag, n, A, x, incX);
    }
}


} // extern "C"
