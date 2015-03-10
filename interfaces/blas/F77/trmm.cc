#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(strmm)(const char     *side_,
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
    float alpha   = *alpha_;
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
        F77BLAS(xerbla)("STRMM ", &info);
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!transA) {
                ulmBLAS::trlmm(m, n, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trumm(m, n, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, 1, ldB);
            }
        } else {
            if (!transA) {
                ulmBLAS::trumm(m, n, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trlmm(m, n, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!transA) {
                ulmBLAS::trumm(n, m, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trlmm(n, m, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        } else {
            if (!transA) {
                ulmBLAS::trlmm(n, m, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trumm(n, m, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        }
    }
}


void
F77BLAS(dtrmm)(const char     *side_,
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
        F77BLAS(xerbla)("DTRMM ", &info);
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!transA) {
                ulmBLAS::trlmm(m, n, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trumm(m, n, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, 1, ldB);
            }
        } else {
            if (!transA) {
                ulmBLAS::trumm(m, n, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trlmm(m, n, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!transA) {
                ulmBLAS::trumm(n, m, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trlmm(n, m, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        } else {
            if (!transA) {
                ulmBLAS::trlmm(n, m, alpha,
                               false, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trumm(n, m, alpha,
                               false, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        }
    }
}

void
F77BLAS(ctrmm)(const char     *side_,
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

    fcomplex  alpha(alpha_[0], alpha_[1]);

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
        F77BLAS(xerbla)("CTRMM ", &info);
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!transA) {
                ulmBLAS::trlmm(m, n, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trumm(m, n, alpha,
                               conjA, unitDiag,
                               A, ldA, 1,
                               B, 1, ldB);
            }
        } else {
            if (!transA) {
                ulmBLAS::trumm(m, n, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trlmm(m, n, alpha, conjA, unitDiag,
                               A, ldA, 1,
                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!transA) {
                ulmBLAS::trumm(n, m, alpha,
                               conjA, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trlmm(n, m, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        } else {
            if (!transA) {
                ulmBLAS::trlmm(n, m, alpha,
                               conjA, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trumm(n, m, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        }
    }
}


void
F77BLAS(ztrmm)(const char     *side_,
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

    dcomplex  alpha(alpha_[0], alpha_[1]);

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
        F77BLAS(xerbla)("ZTRMM ", &info);
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!transA) {
                ulmBLAS::trlmm(m, n, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trumm(m, n, alpha,
                               conjA, unitDiag,
                               A, ldA, 1,
                               B, 1, ldB);
            }
        } else {
            if (!transA) {
                ulmBLAS::trumm(m, n, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, 1, ldB);
            } else {
                ulmBLAS::trlmm(m, n, alpha, conjA, unitDiag,
                               A, ldA, 1,
                               B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!transA) {
                ulmBLAS::trumm(n, m, alpha,
                               conjA, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trlmm(n, m, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        } else {
            if (!transA) {
                ulmBLAS::trlmm(n, m, alpha,
                               conjA, unitDiag, A, ldA, 1,
                               B, ldB, 1);
            } else {
                ulmBLAS::trumm(n, m, alpha,
                               conjA, unitDiag, A, 1, ldA,
                               B, ldB, 1);
            }
        }
    }
}

} // extern "C"
