#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

#include <iostream>

extern "C" {

void
ULMBLAS(stpmv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               enum CBLAS_DIAG       diag,
               ULMBLAS_INDEX_TYPE    n,
               const float           *A,
               float                 *x,
               ULMBLAS_INDEX_TYPE    incX)
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
            ulmBLAS::tplmv<std::ptrdiff_t>(n, unitDiag, A, x, incX);
        } else {
            ulmBLAS::tpumv<std::ptrdiff_t>(n, unitDiag, A, x, incX);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::tplmtv<std::ptrdiff_t>(n, unitDiag, A, x, incX);
        } else {
            ulmBLAS::tpumtv<std::ptrdiff_t>(n, unitDiag, A, x, incX);
        }
    }
}

void
ULMBLAS(dtpmv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               enum CBLAS_DIAG       diag,
               ULMBLAS_INDEX_TYPE    n,
               const double          *A,
               double                *x,
               ULMBLAS_INDEX_TYPE    incX)
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
            ulmBLAS::tplmv<std::ptrdiff_t>(n, unitDiag, A, x, incX);
        } else {
            ulmBLAS::tpumv<std::ptrdiff_t>(n, unitDiag, A, x, incX);
        }
    } else {
        if (upLo==CblasLower) {
            ulmBLAS::tplmtv<std::ptrdiff_t>(n, unitDiag, A, x, incX);
        } else {
            ulmBLAS::tpumtv<std::ptrdiff_t>(n, unitDiag, A, x, incX);
        }
    }
}

void
ULMBLAS(ctpmv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  transA_,
               enum CBLAS_DIAG       diag,
               ULMBLAS_INDEX_TYPE    n,
               const float           *AP_,
               float                 *x_,
               ULMBLAS_INDEX_TYPE    incX)
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
            ulmBLAS::tplmv<std::ptrdiff_t>(n, unitDiag, conjA, AP, x, incX);
        } else {
            ulmBLAS::tpumv<std::ptrdiff_t>(n, unitDiag, conjA, AP, x, incX);
        }
    } else {
        if (lowerA) {
            ulmBLAS::tplmtv<std::ptrdiff_t>(n, unitDiag, conjA, AP, x, incX);
        } else {
            ulmBLAS::tpumtv<std::ptrdiff_t>(n, unitDiag, conjA, AP, x, incX);
        }
    }
}

void
ULMBLAS(ztpmv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  transA_,
               enum CBLAS_DIAG       diag,
               ULMBLAS_INDEX_TYPE    n,
               const double          *AP_,
               double                *x_,
               ULMBLAS_INDEX_TYPE    incX)
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
            ulmBLAS::tplmv<std::ptrdiff_t>(n, unitDiag, conjA, AP, x, incX);
        } else {
            ulmBLAS::tpumv<std::ptrdiff_t>(n, unitDiag, conjA, AP, x, incX);
        }
    } else {
        if (lowerA) {
            ulmBLAS::tplmtv<std::ptrdiff_t>(n, unitDiag, conjA, AP, x, incX);
        } else {
            ulmBLAS::tpumtv<std::ptrdiff_t>(n, unitDiag, conjA, AP, x, incX);
        }
    }
}

void
CBLAS(stpmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             ULMBLAS_INDEX_TYPE    n,
             const float           *A,
             float                 *x,
             ULMBLAS_INDEX_TYPE    incX)
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
        CBLAS(xerbla)(info, "cblas_stpmv", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(stpmv)(upLo, trans, diag, n, A, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(stpmv)(upLo, trans, diag, n, A, x, incX);
    }
}

void
CBLAS(dtpmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             ULMBLAS_INDEX_TYPE    n,
             const double          *A,
             double                *x,
             ULMBLAS_INDEX_TYPE    incX)
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
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(dtpmv)(upLo, trans, diag, n, A, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(dtpmv)(upLo, trans, diag, n, A, x, incX);
    }
}

void
CBLAS(ctpmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             ULMBLAS_INDEX_TYPE    n,
             const float           *A,
             float                 *x,
             ULMBLAS_INDEX_TYPE    incX)
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
        CBLAS(xerbla)(info, "cblas_ctpmv", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(ctpmv)(upLo, trans, diag, n, A, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(ctpmv)(upLo, trans, diag, n, A, x, incX);
    }
}

void
CBLAS(ztpmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             ULMBLAS_INDEX_TYPE    n,
             const double          *A,
             double                *x,
             ULMBLAS_INDEX_TYPE    incX)
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
        CBLAS(xerbla)(info, "cblas_ztpmv", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(ztpmv)(upLo, trans, diag, n, A, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(ztpmv)(upLo, trans, diag, n, A, x, incX);
    }
}

} // extern "C"
