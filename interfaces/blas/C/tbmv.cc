#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

#include <iostream>

extern "C" {

void
ULMBLAS(stbmv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               enum CBLAS_DIAG       diag,
               int                   n,
               int                   k,
               const float           *A,
               int                   ldA,
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
            ulmBLAS::tblmv(n, k, unitDiag, A, ldA, x, incX);
        } else {
            ulmBLAS::tbumv(n, k, unitDiag, A, ldA, x, incX);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::tblmtv(n, k, unitDiag, A, ldA, x, incX);
        } else {
            ulmBLAS::tbumtv(n, k, unitDiag, A, ldA, x, incX);
        }
    }
}

void
ULMBLAS(dtbmv)(enum CBLAS_UPLO       upLo,
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
            ulmBLAS::tblmv(n, k, unitDiag, A, ldA, x, incX);
        } else {
            ulmBLAS::tbumv(n, k, unitDiag, A, ldA, x, incX);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::tblmtv(n, k, unitDiag, A, ldA, x, incX);
        } else {
            ulmBLAS::tbumtv(n, k, unitDiag, A, ldA, x, incX);
        }
    }
}

void
ULMBLAS(ctbmv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  transA_,
               enum CBLAS_DIAG       diag,
               int                   n,
               int                   k,
               const float           *A_,
               int                   ldA,
               float                 *x_,
               int                   incX)
{
    bool lowerA = (upLo==CblasLower);
    bool transA = (transA_==CblasTrans || transA_==CblasConjTrans);
    bool conjA  = (transA_==AtlasConj || transA_==CblasConjTrans);

    typedef std::complex<float> fcomplex;
    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    fcomplex       *x = reinterpret_cast<fcomplex *>(x_);

    if (incX<0) {
        x -= incX*(n-1);
    }

    bool unitDiag = (diag==CblasUnit);

//
//  Start the operations.
//
    if (!transA) {
        if (lowerA) {
            ulmBLAS::tblmv(n, k, unitDiag, conjA, A, ldA, x, incX);
        } else {
            ulmBLAS::tbumv(n, k, unitDiag, conjA, A, ldA, x, incX);
        }
    } else {
        if (lowerA) {
            ulmBLAS::tblmtv(n, k, unitDiag, conjA, A, ldA, x, incX);
        } else {
            ulmBLAS::tbumtv(n, k, unitDiag, conjA, A, ldA, x, incX);
        }
    }
}

void
ULMBLAS(ztbmv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  transA_,
               enum CBLAS_DIAG       diag,
               int                   n,
               int                   k,
               const double          *A_,
               int                   ldA,
               double                *x_,
               int                   incX)
{
    bool lowerA = (upLo==CblasLower);
    bool transA = (transA_==CblasTrans || transA_==CblasConjTrans);
    bool conjA  = (transA_==AtlasConj || transA_==CblasConjTrans);

    typedef std::complex<double> dcomplex;
    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    dcomplex       *x = reinterpret_cast<dcomplex *>(x_);

    if (incX<0) {
        x -= incX*(n-1);
    }

    bool unitDiag = (diag==CblasUnit);

//
//  Start the operations.
//
    if (!transA) {
        if (lowerA) {
            ulmBLAS::tblmv(n, k, unitDiag, conjA, A, ldA, x, incX);
        } else {
            ulmBLAS::tbumv(n, k, unitDiag, conjA, A, ldA, x, incX);
        }
    } else {
        if (lowerA) {
            ulmBLAS::tblmtv(n, k, unitDiag, conjA, A, ldA, x, incX);
        } else {
            ulmBLAS::tbumtv(n, k, unitDiag, conjA, A, ldA, x, incX);
        }
    }
}

void
CBLAS(stbmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             int                   n,
             int                   k,
             const float           *A,
             int                   ldA,
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
        CBLAS(xerbla)(info, "cblas_stbmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(stbmv)(upLo, trans, diag, n, k, A, ldA, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(stbmv)(upLo, trans, diag, n, k, A, ldA, x, incX);
    }
}

void
CBLAS(dtbmv)(enum CBLAS_ORDER      order,
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
        CBLAS(xerbla)(info, "cblas_dtbmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dtbmv)(upLo, trans, diag, n, k, A, ldA, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(dtbmv)(upLo, trans, diag, n, k, A, ldA, x, incX);
    }
}

void
CBLAS(ctbmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             int                   n,
             int                   k,
             const float           *A,
             int                   ldA,
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
        CBLAS(xerbla)(info, "cblas_ctbmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(ctbmv)(upLo, trans, diag, n, k, A, ldA, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(ctbmv)(upLo, trans, diag, n, k, A, ldA, x, incX);
    }
}

void
CBLAS(ztbmv)(enum CBLAS_ORDER      order,
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
        CBLAS(xerbla)(info, "cblas_ztbmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(ztbmv)(upLo, trans, diag, n, k, A, ldA, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(ztbmv)(upLo, trans, diag, n, k, A, ldA, x, incX);
    }
}


} // extern "C"
