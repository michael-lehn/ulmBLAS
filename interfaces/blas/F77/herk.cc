#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(cherk)(const char     *upLo_,
               const char     *trans_,
               const int      *n_,
               const int      *k_,
               const float    *alpha_,
               const float    *A_,
               const int      *ldA_,
               const float    *beta_,
               float          *C_,
               const int      *ldC_)
{
    typedef std::complex<float > fcomplex;
//
//  Dereference scalar parameters
//
    bool trans   = (toupper(*trans_) == 'C');
    bool lower   = (toupper(*upLo_) == 'L');
    int n        = *n_;
    int k        = *k_;
    float alpha  = *alpha_;
    int ldA      = *ldA_;
    float beta   = *beta_;
    int ldC      = *ldC_;

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    fcomplex       *C = reinterpret_cast<fcomplex *>(C_);
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (trans) ? k : n;

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 1;
    } else if (toupper(*trans_)!='N' && toupper(*trans_)!='C')
    {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (k<0) {
        info = 4;
    } else if (ldA<std::max(1,numRowsA)) {
        info = 7;
    } else if (ldC<std::max(1,n)) {
        info = 10;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CHERK ", &info);
    }

//
//  Start the operations.
//
    if (!trans) {
        if (lower) {
            ulmBLAS::helrk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        } else {
            ulmBLAS::heurk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::heurk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::helrk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        }
    }
}


void
F77BLAS(zherk)(const char     *upLo_,
               const char     *trans_,
               const int      *n_,
               const int      *k_,
               const double   *alpha_,
               const double   *A_,
               const int      *ldA_,
               const double   *beta_,
               double         *C_,
               const int      *ldC_)
{
    typedef std::complex<double> dcomplex;
//
//  Dereference scalar parameters
//
    bool trans   = (toupper(*trans_) == 'C');
    bool lower   = (toupper(*upLo_) == 'L');
    int n        = *n_;
    int k        = *k_;
    double alpha = *alpha_;
    int ldA      = *ldA_;
    double beta  = *beta_;
    int ldC      = *ldC_;

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (trans) ? k : n;

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 1;
    } else if (toupper(*trans_)!='N' && toupper(*trans_)!='C')
    {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (k<0) {
        info = 4;
    } else if (ldA<std::max(1,numRowsA)) {
        info = 7;
    } else if (ldC<std::max(1,n)) {
        info = 10;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZHERK ", &info);
    }

//
//  Start the operations.
//
    if (!trans) {
        if (lower) {
            ulmBLAS::helrk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        } else {
            ulmBLAS::heurk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::heurk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::helrk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        }
    }
}

} // extern "C"
