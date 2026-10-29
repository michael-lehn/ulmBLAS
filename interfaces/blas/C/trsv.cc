#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(strsv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               enum CBLAS_DIAG       diag,
               ULMBLAS_INDEX_TYPE    n,
               const float           *A,
               ULMBLAS_INDEX_TYPE    ldA,
               float                 *x,
               ULMBLAS_INDEX_TYPE    incX)
{
//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    bool unitDiag = (diag==CblasUnit);

    if (upLo==CblasLower) {
        if (trans==CblasNoTrans || trans==AtlasConj) {
            ulmBLAS::trlsv<std::ptrdiff_t>(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trusv<std::ptrdiff_t>(n, unitDiag, A, ldA, 1, x, incX);
        }
    } else {
        if (trans==CblasNoTrans || trans==AtlasConj) {
            ulmBLAS::trusv<std::ptrdiff_t>(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trlsv<std::ptrdiff_t>(n, unitDiag, A, ldA, 1, x, incX);
        }
    }
}

void
ULMBLAS(dtrsv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  trans,
               enum CBLAS_DIAG       diag,
               ULMBLAS_INDEX_TYPE    n,
               const double          *A,
               ULMBLAS_INDEX_TYPE    ldA,
               double                *x,
               ULMBLAS_INDEX_TYPE    incX)
{
//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    bool unitDiag = (diag==CblasUnit);

    if (upLo==CblasLower) {
        if (trans==CblasNoTrans || trans==AtlasConj) {
            ulmBLAS::trlsv<std::ptrdiff_t>(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trusv<std::ptrdiff_t>(n, unitDiag, A, ldA, 1, x, incX);
        }
    } else {
        if (trans==CblasNoTrans || trans==AtlasConj) {
            ulmBLAS::trusv<std::ptrdiff_t>(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trlsv<std::ptrdiff_t>(n, unitDiag, A, ldA, 1, x, incX);
        }
    }
}

void
ULMBLAS(ctrsv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  transA_,
               enum CBLAS_DIAG       diag,
               ULMBLAS_INDEX_TYPE    n,
               const float           *A_,
               ULMBLAS_INDEX_TYPE    ldA,
               float                 *x_,
               ULMBLAS_INDEX_TYPE    incX)
{
    bool lower  = (upLo==CblasLower);
    bool transA = (transA_==CblasTrans || transA_==CblasConjTrans);
    bool conjA  = (transA_==AtlasConj || transA_==CblasConjTrans);

    typedef std::complex<float> fcomplex;
    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    fcomplex       *x = reinterpret_cast<fcomplex *>(x_);

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    bool unitDiag = (diag==CblasUnit);

    if (lower) {
        if (!transA) {
            ulmBLAS::trlsv<std::ptrdiff_t>(n, unitDiag, conjA, A, 1, ldA,
                                           x, incX);
        } else {
            ulmBLAS::trusv<std::ptrdiff_t>(n, unitDiag, conjA, A, ldA, 1,
                                           x, incX);
        }
    } else {
        if (!transA) {
            ulmBLAS::trusv<std::ptrdiff_t>(n, unitDiag, conjA, A, 1, ldA,
                                           x, incX);
        } else {
            ulmBLAS::trlsv<std::ptrdiff_t>(n, unitDiag, conjA, A, ldA, 1,
                                           x, incX);
        }
    }
}

void
ULMBLAS(ztrsv)(enum CBLAS_UPLO       upLo,
               enum CBLAS_TRANSPOSE  transA_,
               enum CBLAS_DIAG       diag,
               ULMBLAS_INDEX_TYPE    n,
               const double          *A_,
               ULMBLAS_INDEX_TYPE    ldA,
               double                *x_,
               ULMBLAS_INDEX_TYPE    incX)
{
    bool lower  = (upLo==CblasLower);
    bool transA = (transA_==CblasTrans || transA_==CblasConjTrans);
    bool conjA  = (transA_==AtlasConj || transA_==CblasConjTrans);

    typedef std::complex<double> dcomplex;
    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    dcomplex       *x = reinterpret_cast<dcomplex *>(x_);

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    bool unitDiag = (diag==CblasUnit);

    if (lower) {
        if (!transA) {
            ulmBLAS::trlsv<std::ptrdiff_t>(n, unitDiag, conjA, A, 1, ldA,
                                           x, incX);
        } else {
            ulmBLAS::trusv<std::ptrdiff_t>(n, unitDiag, conjA, A, ldA, 1,
                                           x, incX);
        }
    } else {
        if (!transA) {
            ulmBLAS::trusv<std::ptrdiff_t>(n, unitDiag, conjA, A, 1, ldA,
                                           x, incX);
        } else {
            ulmBLAS::trlsv<std::ptrdiff_t>(n, unitDiag, conjA, A, ldA, 1,
                                           x, incX);
        }
    }
}

void
CBLAS(strsv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             ULMBLAS_INDEX_TYPE    n,
             const float           *A,
             ULMBLAS_INDEX_TYPE    ldA,
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
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_strsv", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(strsv)(upLo, trans, diag, n, A, ldA, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(strsv)(upLo, trans, diag, n, A, ldA, x, incX);
    }
}

void
CBLAS(dtrsv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             ULMBLAS_INDEX_TYPE    n,
             const double          *A,
             ULMBLAS_INDEX_TYPE    ldA,
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
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_dtrsv", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(dtrsv)(upLo, trans, diag, n, A, ldA, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(dtrsv)(upLo, trans, diag, n, A, ldA, x, incX);
    }
}

void
CBLAS(ctrsv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             ULMBLAS_INDEX_TYPE    n,
             const float           *A,
             ULMBLAS_INDEX_TYPE    ldA,
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
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_ctrsv", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(ctrsv)(upLo, trans, diag, n, A, ldA, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(ctrsv)(upLo, trans, diag, n, A, ldA, x, incX);
    }
}

void
CBLAS(ztrsv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             enum CBLAS_TRANSPOSE  trans,
             enum CBLAS_DIAG       diag,
             ULMBLAS_INDEX_TYPE    n,
             const double          *A,
             ULMBLAS_INDEX_TYPE    ldA,
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
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_ztrsv", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(ztrsv)(upLo, trans, diag, n, A, ldA, x, incX);
    } else {
        upLo  = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        trans = transpose(trans);
        ULMBLAS(ztrsv)(upLo, trans, diag, n, A, ldA, x, incX);
    }
}

} // extern "C"
