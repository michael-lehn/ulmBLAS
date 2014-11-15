#include <algorithm>
#include <cctype>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/level3/sylrk.h>
#include <ulmblas/level3/syurk.h>

extern "C" {

void
F77BLAS(dsyrk)(const char     *upLo_,
               const char     *trans_,
               const int      *n_,
               const int      *k_,
               const double   *alpha_,
               const double   *A,
               const int      *ldA_,
               const double   *beta_,
               double         *C,
               const int      *ldC_)
{
//
//  Dereference scalar parameters
//
    bool trans   = (toupper(*trans_) == 'T' || toupper(*trans_) == 'C');
    bool lower   = (toupper(*upLo_) == 'L');
    int n        = *n_;
    int k        = *k_;
    double alpha = *alpha_;
    int ldA      = *ldA_;
    double beta  = *beta_;
    int ldC      = *ldC_;

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
    } else if (toupper(*trans_)!='N'
            && toupper(*trans_)!='T'
            && toupper(*trans_)!='C'
            && toupper(*trans_)!='R')
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
        F77BLAS(xerbla)("DSYRK ", &info);
    }

//
//  Start the operations.
//
    if (!trans) {
        if (lower) {
            ulmBLAS::sylrk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        } else {
            ulmBLAS::syurk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::syurk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::sylrk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        }
    }
}

} // extern "C"
