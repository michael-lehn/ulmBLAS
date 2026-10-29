#include BLAS_HEADER
#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

#include <ulmblas/auxiliary/printmatrix.h>

extern "C" {

void
ULMBLAS(strmm)(enum CBLAS_SIDE      side,
               enum CBLAS_UPLO      upLo,
               enum CBLAS_TRANSPOSE transA,
               enum CBLAS_DIAG      diag,
               ULMBLAS_INDEX_TYPE   m,
               ULMBLAS_INDEX_TYPE   n,
               float                alpha,
               const float          *A,
               ULMBLAS_INDEX_TYPE   ldA,
               float                *B,
               ULMBLAS_INDEX_TYPE   ldB)
{
    bool left     = (side==CblasLeft);
    bool lower    = (upLo==CblasLower);
    bool trans    = (transA==CblasTrans || transA==CblasConjTrans);
    bool unitDiag = (diag==CblasUnit);

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!trans) {
                ulmBLAS::trlmm<std::ptrdiff_t>(m, n, alpha, false, unitDiag,
                                               A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trumm<std::ptrdiff_t>(m, n, alpha, false, unitDiag,
                                               A, ldA, 1,
                                               B, 1, ldB);
            }
        } else {
            if (!trans) {
                ulmBLAS::trumm<std::ptrdiff_t>(m, n, alpha, false, unitDiag,
                                               A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trlmm<std::ptrdiff_t>(m, n, alpha, false, unitDiag,
                                               A, ldA, 1,
                                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!trans) {
                ulmBLAS::trumm<std::ptrdiff_t>(n, m, alpha, false, unitDiag,
                                               A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trlmm<std::ptrdiff_t>(n, m, alpha, false, unitDiag,
                                               A, 1, ldA,
                                               B, ldB, 1);
            }
        } else {
            if (!trans) {
                ulmBLAS::trlmm<std::ptrdiff_t>(n, m, alpha, false, unitDiag,
                                               A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trumm<std::ptrdiff_t>(n, m, alpha, false, unitDiag,
                                               A, 1, ldA,
                                               B, ldB, 1);
            }
        }
    }
}

void
ULMBLAS(dtrmm)(enum CBLAS_SIDE      side,
               enum CBLAS_UPLO      upLo,
               enum CBLAS_TRANSPOSE transA,
               enum CBLAS_DIAG      diag,
               ULMBLAS_INDEX_TYPE   m,
               ULMBLAS_INDEX_TYPE   n,
               double               alpha,
               const double         *A,
               ULMBLAS_INDEX_TYPE   ldA,
               double               *B,
               ULMBLAS_INDEX_TYPE   ldB)
{
    bool left     = (side==CblasLeft);
    bool lower    = (upLo==CblasLower);
    bool trans    = (transA==CblasTrans || transA==CblasConjTrans);
    bool unitDiag = (diag==CblasUnit);

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!trans) {
                ulmBLAS::trlmm<std::ptrdiff_t>(m, n, alpha, false, unitDiag,
                                               A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trumm<std::ptrdiff_t>(m, n, alpha, false, unitDiag,
                                               A, ldA, 1,
                                               B, 1, ldB);
            }
        } else {
            if (!trans) {
                ulmBLAS::trumm<std::ptrdiff_t>(m, n, alpha, false, unitDiag,
                                               A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trlmm<std::ptrdiff_t>(m, n, alpha, false, unitDiag,
                                               A, ldA, 1,
                                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!trans) {
                ulmBLAS::trumm<std::ptrdiff_t>(n, m, alpha, false, unitDiag,
                                               A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trlmm<std::ptrdiff_t>(n, m, alpha, false, unitDiag,
                                               A, 1, ldA,
                                               B, ldB, 1);
            }
        } else {
            if (!trans) {
                ulmBLAS::trlmm<std::ptrdiff_t>(n, m, alpha, false, unitDiag,
                                               A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trumm<std::ptrdiff_t>(n, m, alpha, false, unitDiag,
                                               A, 1, ldA,
                                               B, ldB, 1);
            }
        }
    }
}

void
ULMBLAS(ctrmm)(enum CBLAS_SIDE      side,
               enum CBLAS_UPLO      upLo,
               enum CBLAS_TRANSPOSE transA,
               enum CBLAS_DIAG      diag,
               ULMBLAS_INDEX_TYPE   m,
               ULMBLAS_INDEX_TYPE   n,
               const float          *alpha_,
               const float          *A_,
               ULMBLAS_INDEX_TYPE   ldA,
               float                *B_,
               ULMBLAS_INDEX_TYPE   ldB)
{
    typedef std::complex<float> fcomplex;

    bool left     = (side==CblasLeft);
    bool lower    = (upLo==CblasLower);
    bool trans    = (transA==CblasTrans || transA==CblasConjTrans);
    bool conjA    = (transA==AtlasConj || transA==CblasConjTrans);
    bool unitDiag = (diag==CblasUnit);

    fcomplex alpha(alpha_[0], alpha_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    fcomplex       *B = reinterpret_cast<fcomplex *>(B_);

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!trans) {
                ulmBLAS::trlmm<std::ptrdiff_t>(m, n, alpha, conjA, unitDiag,
                                               A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trumm<std::ptrdiff_t>(m, n, alpha, conjA, unitDiag,
                                               A, ldA, 1,
                                               B, 1, ldB);
            }
        } else {
            if (!trans) {
                ulmBLAS::trumm<std::ptrdiff_t>(m, n, alpha, conjA, unitDiag,
                                               A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trlmm<std::ptrdiff_t>(m, n, alpha, conjA, unitDiag,
                                               A, ldA, 1,
                                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!trans) {
                ulmBLAS::trumm<std::ptrdiff_t>(n, m, alpha, conjA, unitDiag,
                                               A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trlmm<std::ptrdiff_t>(n, m, alpha, conjA, unitDiag,
                                               A, 1, ldA,
                                               B, ldB, 1);
            }
        } else {
            if (!trans) {
                ulmBLAS::trlmm<std::ptrdiff_t>(n, m, alpha, conjA, unitDiag,
                                               A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trumm<std::ptrdiff_t>(n, m, alpha, conjA, unitDiag,
                                               A, 1, ldA,
                                               B, ldB, 1);
            }
        }
    }
}

void
ULMBLAS(ztrmm)(enum CBLAS_SIDE      side,
               enum CBLAS_UPLO      upLo,
               enum CBLAS_TRANSPOSE transA,
               enum CBLAS_DIAG      diag,
               ULMBLAS_INDEX_TYPE   m,
               ULMBLAS_INDEX_TYPE   n,
               const double         *alpha_,
               const double         *A_,
               ULMBLAS_INDEX_TYPE   ldA,
               double               *B_,
               ULMBLAS_INDEX_TYPE   ldB)
{
    typedef std::complex<double> dcomplex;

    bool left     = (side==CblasLeft);
    bool lower    = (upLo==CblasLower);
    bool trans    = (transA==CblasTrans || transA==CblasConjTrans);
    bool conjA    = (transA==AtlasConj || transA==CblasConjTrans);
    bool unitDiag = (diag==CblasUnit);

    dcomplex alpha(alpha_[0], alpha_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    dcomplex       *B = reinterpret_cast<dcomplex *>(B_);

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!trans) {
                ulmBLAS::trlmm<std::ptrdiff_t>(m, n, alpha, conjA, unitDiag,
                                               A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trumm<std::ptrdiff_t>(m, n, alpha, conjA, unitDiag,
                                               A, ldA, 1,
                                               B, 1, ldB);
            }
        } else {
            if (!trans) {
                ulmBLAS::trumm<std::ptrdiff_t>(m, n, alpha, conjA, unitDiag,
                                               A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trlmm<std::ptrdiff_t>(m, n, alpha, conjA, unitDiag,
                                               A, ldA, 1,
                                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!trans) {
                ulmBLAS::trumm<std::ptrdiff_t>(n, m, alpha, conjA, unitDiag,
                                               A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trlmm<std::ptrdiff_t>(n, m, alpha, conjA, unitDiag,
                                               A, 1, ldA,
                                               B, ldB, 1);
            }
        } else {
            if (!trans) {
                ulmBLAS::trlmm<std::ptrdiff_t>(n, m, alpha, conjA, unitDiag,
                                               A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trumm<std::ptrdiff_t>(n, m, alpha, conjA, unitDiag,
                                               A, 1, ldA,
                                               B, ldB, 1);
            }
        }
    }
}

void
CBLAS(strmm)(enum CBLAS_ORDER         order,
             enum CBLAS_SIDE          side,
             enum CBLAS_UPLO          upLo,
             enum CBLAS_TRANSPOSE     transA,
             enum CBLAS_DIAG          diag,
             ULMBLAS_INDEX_TYPE       m,
             ULMBLAS_INDEX_TYPE       n,
             float                    alpha,
             const float              *A,
             ULMBLAS_INDEX_TYPE       ldA,
             float                    *B,
             const ULMBLAS_INDEX_TYPE ldB)
{
    bool left     = (side==CblasLeft);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (side!=CblasLeft && side!=CblasRight) {
        info = 2;
    } else if (upLo!=CblasLower && upLo!=CblasUpper) {
        info = 3;
    } else if (transA!=CblasNoTrans && transA!=CblasTrans
            && transA!=AtlasConj && transA!=CblasConjTrans)
    {
        info = 4;
    } else if (diag!=CblasUnit && diag!=CblasNonUnit) {
        info = 5;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 6;
            } else if (n<0) {
                info = 7;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 10;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
                info = 12;
            }
        } else {
            if (n<0) {
                info = 6;
            } else if (m<0) {
                info = 7;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 10;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
                info = 12;
            }
        }
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_strmm", "");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(strmm)(side, upLo, transA, diag, m, n, alpha, A, ldA, B, ldB);
    } else {
        side   = (side==CblasLeft)  ? CblasRight : CblasLeft;
        upLo   = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(strmm)(side, upLo, transA, diag, n, m, alpha, A, ldA, B, ldB);
    }
}


void
CBLAS(dtrmm)(enum CBLAS_ORDER         order,
             enum CBLAS_SIDE          side,
             enum CBLAS_UPLO          upLo,
             enum CBLAS_TRANSPOSE     transA,
             enum CBLAS_DIAG          diag,
             ULMBLAS_INDEX_TYPE       m,
             ULMBLAS_INDEX_TYPE       n,
             double                   alpha,
             const double             *A,
             ULMBLAS_INDEX_TYPE       ldA,
             double                   *B,
             const ULMBLAS_INDEX_TYPE ldB)
{
    bool left     = (side==CblasLeft);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (side!=CblasLeft && side!=CblasRight) {
        info = 2;
    } else if (upLo!=CblasLower && upLo!=CblasUpper) {
        info = 3;
    } else if (transA!=CblasNoTrans && transA!=CblasTrans
            && transA!=AtlasConj && transA!=CblasConjTrans)
    {
        info = 4;
    } else if (diag!=CblasUnit && diag!=CblasNonUnit) {
        info = 5;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 6;
            } else if (n<0) {
                info = 7;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 10;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
                info = 12;
            }
        } else {
            if (n<0) {
                info = 6;
            } else if (m<0) {
                info = 7;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 10;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
                info = 12;
            }
        }
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_dtrmm", "");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(dtrmm)(side, upLo, transA, diag, m, n, alpha, A, ldA, B, ldB);
    } else {
        side   = (side==CblasLeft)  ? CblasRight : CblasLeft;
        upLo   = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(dtrmm)(side, upLo, transA, diag, n, m, alpha, A, ldA, B, ldB);
    }
}

void
CBLAS(ctrmm)(enum CBLAS_ORDER         order,
             enum CBLAS_SIDE          side,
             enum CBLAS_UPLO          upLo,
             enum CBLAS_TRANSPOSE     transA,
             enum CBLAS_DIAG          diag,
             ULMBLAS_INDEX_TYPE       m,
             ULMBLAS_INDEX_TYPE       n,
             const float              *alpha,
             const float              *A,
             ULMBLAS_INDEX_TYPE       ldA,
             float                    *B,
             const ULMBLAS_INDEX_TYPE ldB)
{
    bool left     = (side==CblasLeft);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (side!=CblasLeft && side!=CblasRight) {
        info = 2;
    } else if (upLo!=CblasLower && upLo!=CblasUpper) {
        info = 3;
    } else if (transA!=CblasNoTrans && transA!=CblasTrans
            && transA!=AtlasConj && transA!=CblasConjTrans)
    {
        info = 4;
    } else if (diag!=CblasUnit && diag!=CblasNonUnit) {
        info = 5;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 6;
            } else if (n<0) {
                info = 7;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 10;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
                info = 12;
            }
        } else {
            if (n<0) {
                info = 6;
            } else if (m<0) {
                info = 7;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 10;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
                info = 12;
            }
        }
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_ctrmm", "");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(ctrmm)(side, upLo, transA, diag, m, n, alpha, A, ldA, B, ldB);
    } else {
        side   = (side==CblasLeft)  ? CblasRight : CblasLeft;
        upLo   = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(ctrmm)(side, upLo, transA, diag, n, m, alpha, A, ldA, B, ldB);
    }
}

void
CBLAS(ztrmm)(enum CBLAS_ORDER         order,
             enum CBLAS_SIDE          side,
             enum CBLAS_UPLO          upLo,
             enum CBLAS_TRANSPOSE     transA,
             enum CBLAS_DIAG          diag,
             ULMBLAS_INDEX_TYPE       m,
             ULMBLAS_INDEX_TYPE       n,
             const double             *alpha,
             const double             *A,
             ULMBLAS_INDEX_TYPE       ldA,
             double                   *B,
             const ULMBLAS_INDEX_TYPE ldB)
{
    bool left     = (side==CblasLeft);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (side!=CblasLeft && side!=CblasRight) {
        info = 2;
    } else if (upLo!=CblasLower && upLo!=CblasUpper) {
        info = 3;
    } else if (transA!=CblasNoTrans && transA!=CblasTrans
            && transA!=AtlasConj && transA!=CblasConjTrans)
    {
        info = 4;
    } else if (diag!=CblasUnit && diag!=CblasNonUnit) {
        info = 5;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 6;
            } else if (n<0) {
                info = 7;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 10;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
                info = 12;
            }
        } else {
            if (n<0) {
                info = 6;
            } else if (m<0) {
                info = 7;
            } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
                info = 10;
            } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
                info = 12;
            }
        }
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_ztrmm", "");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(ztrmm)(side, upLo, transA, diag, m, n, alpha, A, ldA, B, ldB);
    } else {
        side   = (side==CblasLeft)  ? CblasRight : CblasLeft;
        upLo   = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(ztrmm)(side, upLo, transA, diag, n, m, alpha, A, ldA, B, ldB);
    }
}

} // extern "C"
