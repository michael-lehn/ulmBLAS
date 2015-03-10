#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(strmv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               enum CBLAS_DIAG       diag,
               int                   n,
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
    if (upLo==CblasLower) {
        if (trans==CblasNoTrans || trans==AtlasConj) {
            ulmBLAS::trlmv(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trumv(n, unitDiag, A, ldA, 1, x, incX);
        }
    } else {
        if (trans==CblasNoTrans || trans==AtlasConj) {
            ulmBLAS::trumv(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trlmv(n, unitDiag, A, ldA, 1, x, incX);
        }
    }
}


void
ULMBLAS(dtrmv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               enum CBLAS_DIAG       diag,
               int                   n,
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
    if (upLo==CblasLower) {
        if (trans==CblasNoTrans || trans==AtlasConj) {
            ulmBLAS::trlmv(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trumv(n, unitDiag, A, ldA, 1, x, incX);
        }
    } else {
        if (trans==CblasNoTrans || trans==AtlasConj) {
            ulmBLAS::trumv(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trlmv(n, unitDiag, A, ldA, 1, x, incX);
        }
    }
}

void
ULMBLAS(ctrmv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  transA_,
               enum CBLAS_DIAG       diag,
               int                   n,
               const float           *A_,
               int                   ldA,
               float                 *x_,
               int                   incX)
{
    bool lower  = (upLo==CblasLower);
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
    if (lower) {
        if (!transA) {
            ulmBLAS::trlmv(n, unitDiag, conjA, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trumv(n, unitDiag, conjA, A, ldA, 1, x, incX);
        }
    } else {
        if (!transA) {
            ulmBLAS::trumv(n, unitDiag, conjA, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trlmv(n, unitDiag, conjA, A, ldA, 1, x, incX);
        }
    }
}

void
ULMBLAS(ztrmv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  transA_,
               enum CBLAS_DIAG       diag,
               int                   n,
               const double          *A_,
               int                   ldA,
               double                *x_,
               int                   incX)
{
    bool lower  = (upLo==CblasLower);
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
    if (lower) {
        if (!transA) {
            ulmBLAS::trlmv(n, unitDiag, conjA, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trumv(n, unitDiag, conjA, A, ldA, 1, x, incX);
        }
    } else {
        if (!transA) {
            ulmBLAS::trumv(n, unitDiag, conjA, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trlmv(n, unitDiag, conjA, A, ldA, 1, x, incX);
        }
    }
}

void
CBLAS(strmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             int                   n,
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
    } else if (ldA<std::max(1,n)) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_strmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(strmv)(upLo, trans, diag, n, A, ldA, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(strmv)(upLo, trans, diag, n, A, ldA, x, incX);
    }
}

void
CBLAS(dtrmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             int                   n,
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
    } else if (ldA<std::max(1,n)) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dtrmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dtrmv)(upLo, trans, diag, n, A, ldA, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(dtrmv)(upLo, trans, diag, n, A, ldA, x, incX);
    }
}

void
CBLAS(ctrmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             int                   n,
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
    } else if (ldA<std::max(1,n)) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_ctrmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(ctrmv)(upLo, trans, diag, n, A, ldA, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(ctrmv)(upLo, trans, diag, n, A, ldA, x, incX);
    }
}

void
CBLAS(ztrmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             int                   n,
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
    } else if (ldA<std::max(1,n)) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_ztrmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(ztrmv)(upLo, trans, diag, n, A, ldA, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(ztrmv)(upLo, trans, diag, n, A, ldA, x, incX);
    }
}

} // extern "C"
