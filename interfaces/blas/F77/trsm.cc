#include <algorithm>
#include <cctype>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(strsm)(const char               *side_,
               const char               *upLo_,
               const char               *transA_,
               const char               *diag_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               const float              *A,
               const ULMBLAS_INDEX_TYPE *ldA_,
               float                    *B,
               const ULMBLAS_INDEX_TYPE *ldB_)
{
//
//  Dereference scalar parameters
//
    bool left              = (toupper(*side_) == 'L');
    bool lower             = (toupper(*upLo_) == 'L');
    bool transA            = (toupper(*transA_) == 'T'
                           || toupper(*transA_) == 'C');
    bool unitDiag          = (toupper(*diag_) == 'U');
    ULMBLAS_INDEX_TYPE m   = *m_;
    ULMBLAS_INDEX_TYPE n   = *n_;
    float  alpha           = *alpha_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*side_)!='L' && toupper(*side_)!='R') {
        info = 1;
    } else if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 2;
    } else if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
            && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 3;
    } else if (toupper(*diag_)!='U' && toupper(*diag_)!='N') {
        info = 4;
    } else if (m<0) {
        info = 5;
    } else if (n<0) {
        info = 6;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 9;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("STRSM ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!transA) {
                ulmBLAS::trlsm<std::ptrdiff_t>(m, n, alpha,
                                               false, unitDiag, A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trusm<std::ptrdiff_t>(m, n, alpha,
                                               false, unitDiag, A, ldA, 1,
                                               B, 1, ldB);
            }
        } else {
            if (!transA) {
                ulmBLAS::trusm<std::ptrdiff_t>(m, n, alpha,
                                               false, unitDiag, A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trlsm<std::ptrdiff_t>(m, n, alpha,
                                               false, unitDiag, A, ldA, 1,
                                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!transA) {
                ulmBLAS::trusm<std::ptrdiff_t>(n, m, alpha,
                                               false, unitDiag, A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trlsm<std::ptrdiff_t>(n, m, alpha,
                                               false, unitDiag, A, 1, ldA,
                                               B, ldB, 1);
            }
        } else {
            if (!transA) {
                ulmBLAS::trlsm<std::ptrdiff_t>(n, m, alpha,
                                               false, unitDiag, A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trusm<std::ptrdiff_t>(n, m, alpha,
                                               false, unitDiag, A, 1, ldA,
                                               B, ldB, 1);
            }
        }
    }
}

void
F77BLAS(dtrsm)(const char               *side_,
               const char               *upLo_,
               const char               *transA_,
               const char               *diag_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const double             *alpha_,
               const double             *A,
               const ULMBLAS_INDEX_TYPE *ldA_,
               double                   *B,
               const ULMBLAS_INDEX_TYPE *ldB_)
{
//
//  Dereference scalar parameters
//
    bool left              = (toupper(*side_) == 'L');
    bool lower             = (toupper(*upLo_) == 'L');
    bool transA            = (toupper(*transA_) == 'T'
                           || toupper(*transA_) == 'C');
    bool unitDiag          = (toupper(*diag_) == 'U');
    ULMBLAS_INDEX_TYPE m   = *m_;
    ULMBLAS_INDEX_TYPE n   = *n_;
    double alpha           = *alpha_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*side_)!='L' && toupper(*side_)!='R') {
        info = 1;
    } else if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 2;
    } else if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
            && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 3;
    } else if (toupper(*diag_)!='U' && toupper(*diag_)!='N') {
        info = 4;
    } else if (m<0) {
        info = 5;
    } else if (n<0) {
        info = 6;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 9;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DTRSM ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!transA) {
                ulmBLAS::trlsm<std::ptrdiff_t>(m, n, alpha,
                                               false, unitDiag, A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trusm<std::ptrdiff_t>(m, n, alpha,
                                               false, unitDiag, A, ldA, 1,
                                               B, 1, ldB);
            }
        } else {
            if (!transA) {
                ulmBLAS::trusm<std::ptrdiff_t>(m, n, alpha,
                                               false, unitDiag, A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trlsm<std::ptrdiff_t>(m, n, alpha,
                                               false, unitDiag, A, ldA, 1,
                                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!transA) {
                ulmBLAS::trusm<std::ptrdiff_t>(n, m, alpha,
                                               false, unitDiag, A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trlsm<std::ptrdiff_t>(n, m, alpha,
                                               false, unitDiag, A, 1, ldA,
                                               B, ldB, 1);
            }
        } else {
            if (!transA) {
                ulmBLAS::trlsm<std::ptrdiff_t>(n, m, alpha,
                                               false, unitDiag, A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trusm<std::ptrdiff_t>(n, m, alpha,
                                               false, unitDiag, A, 1, ldA,
                                               B, ldB, 1);
            }
        }
    }
}

void
F77BLAS(ctrsm)(const char               *side_,
               const char               *upLo_,
               const char               *transA_,
               const char               *diag_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               const float              *A_,
               const ULMBLAS_INDEX_TYPE *ldA_,
               float                    *B_,
               const ULMBLAS_INDEX_TYPE *ldB_)
{
    typedef std::complex<float> fcomplex;

//
//  Dereference scalar parameters
//
    bool left              = (toupper(*side_) == 'L');
    bool lower             = (toupper(*upLo_) == 'L');
    bool transA            = (toupper(*transA_) == 'T'
                           || toupper(*transA_) == 'C');
    bool conjA             = (toupper(*transA_) == 'R'
                           || toupper(*transA_) == 'C');
    bool unitDiag          = (toupper(*diag_) == 'U');
    ULMBLAS_INDEX_TYPE m   = *m_;
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;

    fcomplex alpha(alpha_[0], alpha_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    fcomplex       *B = reinterpret_cast<fcomplex *>(B_);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*side_)!='L' && toupper(*side_)!='R') {
        info = 1;
    } else if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 2;
    } else if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
            && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 3;
    } else if (toupper(*diag_)!='U' && toupper(*diag_)!='N') {
        info = 4;
    } else if (m<0) {
        info = 5;
    } else if (n<0) {
        info = 6;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 9;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CTRSM ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!transA) {
                ulmBLAS::trlsm<std::ptrdiff_t>(m, n, alpha,
                                               conjA, unitDiag, A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trusm<std::ptrdiff_t>(m, n, alpha,
                                               conjA, unitDiag, A, ldA, 1,
                                               B, 1, ldB);
            }
        } else {
            if (!transA) {
                ulmBLAS::trusm<std::ptrdiff_t>(m, n, alpha,
                                               conjA, unitDiag, A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trlsm<std::ptrdiff_t>(m, n, alpha,
                                               conjA, unitDiag, A, ldA, 1,
                                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!transA) {
                ulmBLAS::trusm<std::ptrdiff_t>(n, m, alpha,
                                               conjA, unitDiag, A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trlsm<std::ptrdiff_t>(n, m, alpha,
                                               conjA, unitDiag, A, 1, ldA,
                                               B, ldB, 1);
            }
        } else {
            if (!transA) {
                ulmBLAS::trlsm<std::ptrdiff_t>(n, m, alpha,
                                               conjA, unitDiag, A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trusm<std::ptrdiff_t>(n, m, alpha,
                                               conjA, unitDiag, A, 1, ldA,
                                               B, ldB, 1);
            }
        }
    }
}


void
F77BLAS(ztrsm)(const char               *side_,
               const char               *upLo_,
               const char               *transA_,
               const char               *diag_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const double             *alpha_,
               const double             *A_,
               const ULMBLAS_INDEX_TYPE *ldA_,
               double                   *B_,
               const ULMBLAS_INDEX_TYPE *ldB_)
{
    typedef std::complex<double> dcomplex;

//
//  Dereference scalar parameters
//
    bool left              = (toupper(*side_) == 'L');
    bool lower             = (toupper(*upLo_) == 'L');
    bool transA            = (toupper(*transA_) == 'T'
                           || toupper(*transA_) == 'C');
    bool conjA             = (toupper(*transA_) == 'R'
                           || toupper(*transA_) == 'C');
    bool unitDiag          = (toupper(*diag_) == 'U');
    ULMBLAS_INDEX_TYPE m   = *m_;
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;

    dcomplex alpha(alpha_[0], alpha_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    dcomplex       *B = reinterpret_cast<dcomplex *>(B_);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*side_)!='L' && toupper(*side_)!='R') {
        info = 1;
    } else if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 2;
    } else if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
            && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 3;
    } else if (toupper(*diag_)!='U' && toupper(*diag_)!='N') {
        info = 4;
    } else if (m<0) {
        info = 5;
    } else if (n<0) {
        info = 6;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 9;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZTRSM ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!transA) {
                ulmBLAS::trlsm<std::ptrdiff_t>(m, n, alpha,
                                               conjA, unitDiag, A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trusm<std::ptrdiff_t>(m, n, alpha,
                                               conjA, unitDiag, A, ldA, 1,
                                               B, 1, ldB);
            }
        } else {
            if (!transA) {
                ulmBLAS::trusm<std::ptrdiff_t>(m, n, alpha,
                                               conjA, unitDiag, A, 1, ldA,
                                               B, 1, ldB);
            } else {
                ulmBLAS::trlsm<std::ptrdiff_t>(m, n, alpha,
                                               conjA, unitDiag, A, ldA, 1,
                                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!transA) {
                ulmBLAS::trusm<std::ptrdiff_t>(n, m, alpha,
                                               conjA, unitDiag, A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trlsm<std::ptrdiff_t>(n, m, alpha,
                                               conjA, unitDiag, A, 1, ldA,
                                               B, ldB, 1);
            }
        } else {
            if (!transA) {
                ulmBLAS::trlsm<std::ptrdiff_t>(n, m, alpha,
                                               conjA, unitDiag, A, ldA, 1,
                                               B, ldB, 1);
            } else {
                ulmBLAS::trusm<std::ptrdiff_t>(n, m, alpha,
                                               conjA, unitDiag, A, 1, ldA,
                                               B, ldB, 1);
            }
        }
    }
}

} // extern "C"
