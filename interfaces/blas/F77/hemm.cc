#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(chemm)(const char     *side_,
               const char     *upLo_,
               const int      *m_,
               const int      *n_,
               const float    *alpha_,
               const float    *A_,
               const int      *ldA_,
               const float    *B_,
               const int      *ldB_,
               const float    *beta_,
               float          *C_,
               const int      *ldC_)
{
    typedef std::complex<float> fcomplex;
//
//  Dereference scalar parameters
//
    bool left     = (toupper(*side_) == 'L');
    bool lower    = (toupper(*upLo_) == 'L');
    int m         = *m_;
    int n         = *n_;
    int ldA       = *ldA_;
    int ldB       = *ldB_;
    int ldC       = *ldC_;

    fcomplex alpha(alpha_[0], alpha_[1]);
    fcomplex beta(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *B = reinterpret_cast<const fcomplex *>(B_);
    fcomplex       *C = reinterpret_cast<fcomplex *>(C_);

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
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (ldA<std::max(1,numRowsA)) {
        info = 7;
    } else if (ldB<std::max(1,m)) {
        info = 9;
    } else if (ldC<std::max(1,m)) {
        info = 12;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CHEMM ", &info);
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            ulmBLAS::helmm(m, n, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        } else {
            ulmBLAS::heumm(m, n, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::heumm(n, m, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::helmm(n, m, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        }
    }
}


void
F77BLAS(zhemm)(const char     *side_,
               const char     *upLo_,
               const int      *m_,
               const int      *n_,
               const double   *alpha_,
               const double   *A_,
               const int      *ldA_,
               const double   *B_,
               const int      *ldB_,
               const double   *beta_,
               double         *C_,
               const int      *ldC_)
{
    typedef std::complex<double> dcomplex;
//
//  Dereference scalar parameters
//
    bool left     = (toupper(*side_) == 'L');
    bool lower    = (toupper(*upLo_) == 'L');
    int m         = *m_;
    int n         = *n_;
    int ldA       = *ldA_;
    int ldB       = *ldB_;
    int ldC       = *ldC_;

    dcomplex alpha(alpha_[0], alpha_[1]);
    dcomplex beta(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *B = reinterpret_cast<const dcomplex *>(B_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);

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
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (ldA<std::max(1,numRowsA)) {
        info = 7;
    } else if (ldB<std::max(1,m)) {
        info = 9;
    } else if (ldC<std::max(1,m)) {
        info = 12;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZHEMM ", &info);
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            ulmBLAS::helmm(m, n, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        } else {
            ulmBLAS::heumm(m, n, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::heumm(n, m, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::helmm(n, m, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        }
    }
}

} // extern "C"
