#include <algorithm>
#include <cctype>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(strsm)(const char     *side_,
               const char     *upLo_,
               const char     *transA_,
               const char     *diag_,
               const int      *m_,
               const int      *n_,
               const float    *alpha_,
               const float    *A,
               const int      *ldA_,
               float          *B,
               const int      *ldB_)
{
//
//  Dereference scalar parameters
//
    bool left     = (toupper(*side_) == 'L');
    bool lower    = (toupper(*upLo_) == 'L');
    bool transA   = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool unitDiag = (toupper(*diag_) == 'U');
    int m         = *m_;
    int n         = *n_;
    double alpha  = *alpha_;
    int ldA       = *ldA_;
    int ldB       = *ldB_;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

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
    } else if (ldA<std::max(1,numRowsA)) {
        info = 9;
    } else if (ldB<std::max(1,m)) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("STRSM ", &info);
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!transA) {
                ulmBLAS::trlsm(m, n, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trusm(m, n, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, 1, ldB);
            }
        } else {
            if (!transA) {
                ulmBLAS::trusm(m, n, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trlsm(m, n, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!transA) {
                ulmBLAS::trusm(n, m, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trlsm(n, m, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        } else {
            if (!transA) {
                ulmBLAS::trlsm(n, m, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trusm(n, m, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        }
    }
}

void
F77BLAS(dtrsm)(const char     *side_,
               const char     *upLo_,
               const char     *transA_,
               const char     *diag_,
               const int      *m_,
               const int      *n_,
               const double   *alpha_,
               const double   *A,
               const int      *ldA_,
               double         *B,
               const int      *ldB_)
{
//
//  Dereference scalar parameters
//
    bool left     = (toupper(*side_) == 'L');
    bool lower    = (toupper(*upLo_) == 'L');
    bool transA   = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool unitDiag = (toupper(*diag_) == 'U');
    int m         = *m_;
    int n         = *n_;
    double alpha  = *alpha_;
    int ldA       = *ldA_;
    int ldB       = *ldB_;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

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
    } else if (ldA<std::max(1,numRowsA)) {
        info = 9;
    } else if (ldB<std::max(1,m)) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DTRSM ", &info);
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!transA) {
                ulmBLAS::trlsm(m, n, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trusm(m, n, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, 1, ldB);
            }
        } else {
            if (!transA) {
                ulmBLAS::trusm(m, n, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trlsm(m, n, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!transA) {
                ulmBLAS::trusm(n, m, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trlsm(n, m, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        } else {
            if (!transA) {
                ulmBLAS::trlsm(n, m, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trusm(n, m, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        }
    }
}

void
F77BLAS(ctrsm)(const char     *side_,
               const char     *upLo_,
               const char     *transA_,
               const char     *diag_,
               const int      *m_,
               const int      *n_,
               const float    *alpha_,
               const float    *A_,
               const int      *ldA_,
               float          *B_,
               const int      *ldB_)
{
    typedef std::complex<float> fcomplex;

//
//  Dereference scalar parameters
//
    bool left     = (toupper(*side_) == 'L');
    bool lower    = (toupper(*upLo_) == 'L');
    bool transA   = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool conjA    = (toupper(*transA_) == 'R' || toupper(*transA_) == 'C');
    bool unitDiag = (toupper(*diag_) == 'U');
    int m         = *m_;
    int n         = *n_;
    int ldA       = *ldA_;
    int ldB       = *ldB_;

    fcomplex alpha(alpha_[0], alpha_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    fcomplex       *B = reinterpret_cast<fcomplex *>(B_);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

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
    } else if (ldA<std::max(1,numRowsA)) {
        info = 9;
    } else if (ldB<std::max(1,m)) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CTRSM ", &info);
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!transA) {
                ulmBLAS::trlsm(m, n, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trusm(m, n, alpha,
                               conjA, unitDiag, A, ldA, 1,
                               B, 1, ldB);
            }
        } else {
            if (!transA) {
                ulmBLAS::trusm(m, n, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trlsm(m, n, alpha,
                               conjA, unitDiag, A, ldA, 1,
                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!transA) {
                ulmBLAS::trusm(n, m, alpha,
                               conjA, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trlsm(n, m, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        } else {
            if (!transA) {
                ulmBLAS::trlsm(n, m, alpha,
                               conjA, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trusm(n, m, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        }
    }
}


void
F77BLAS(ztrsm)(const char     *side_,
               const char     *upLo_,
               const char     *transA_,
               const char     *diag_,
               const int      *m_,
               const int      *n_,
               const double   *alpha_,
               const double   *A_,
               const int      *ldA_,
               double         *B_,
               const int      *ldB_)
{
    typedef std::complex<double> dcomplex;

//
//  Dereference scalar parameters
//
    bool left     = (toupper(*side_) == 'L');
    bool lower    = (toupper(*upLo_) == 'L');
    bool transA   = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool conjA    = (toupper(*transA_) == 'R' || toupper(*transA_) == 'C');
    bool unitDiag = (toupper(*diag_) == 'U');
    int m         = *m_;
    int n         = *n_;
    int ldA       = *ldA_;
    int ldB       = *ldB_;

    dcomplex alpha(alpha_[0], alpha_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    dcomplex       *B = reinterpret_cast<dcomplex *>(B_);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

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
    } else if (ldA<std::max(1,numRowsA)) {
        info = 9;
    } else if (ldB<std::max(1,m)) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZTRSM ", &info);
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!transA) {
                ulmBLAS::trlsm(m, n, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trusm(m, n, alpha,
                               conjA, unitDiag, A, ldA, 1,
                               B, 1, ldB);
            }
        } else {
            if (!transA) {
                ulmBLAS::trusm(m, n, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trlsm(m, n, alpha,
                               conjA, unitDiag, A, ldA, 1,
                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!transA) {
                ulmBLAS::trusm(n, m, alpha,
                               conjA, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trlsm(n, m, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        } else {
            if (!transA) {
                ulmBLAS::trlsm(n, m, alpha,
                               conjA, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trusm(n, m, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        }
    }
}

} // extern "C"
