#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/F77/config.h>
#include <interfaces/F77/xerbla.h>
#include <ulmblas/level3/sylrk.h>
#include <ulmblas/level3/syurk.h>

extern "C" {

void
F77BLAS(dsyrk)(const char     *_upLo,
               const char     *_trans,
               const int      *_n,
               const int      *_k,
               const double   *_alpha,
               const double   *A,
               const int      *_ldA,
               const double   *_beta,
               double         *C,
               const int      *_ldC)
{
//
//  Dereference scalar parameters
//
    bool trans   = (toupper(*_trans) == 'T' || toupper(*_trans) == 'C');
    bool lower   = (toupper(*_upLo) == 'L');
    int n        = *_n;
    int k        = *_k;
    double alpha = *_alpha;
    int ldA      = *_ldA;
    double beta  = *_beta;
    int ldC      = *_ldC;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (trans) ? k : n;

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*_upLo)!='L' && toupper(*_upLo)!='U') {
        info = 1;
    } else if (toupper(*_trans)!='N'
            && toupper(*_trans)!='T'
            && toupper(*_trans)!='C'
            && toupper(*_trans)!='R')
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
