#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/F77/config.h>
#include <interfaces/F77/xerbla.h>
#include <src/level2/trlmv.h>
#include <src/level2/trumv.h>

extern "C" {

void
F77BLAS(dtrmv)(const char     *_upLo,
               const char     *_transA,
               const char     *_diag,
               const int      *_n,
               const double   *A,
               const int      *_ldA,
               double         *x,
               const int      *_incX)
{
//
//  Dereference scalar parameters
//
    bool lower    = (toupper(*_upLo) == 'L');
    bool transA   = (toupper(*_transA) == 'T' || toupper(*_transA) == 'C');
    bool unitDiag = (toupper(*_diag) == 'U');
    int  n        = *_n;
    int  ldA      = *_ldA;
    int  incX     = *_incX;

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*_upLo)!='U' && toupper(*_upLo)!='L') {
        info = 1;
    } else if (toupper(*_transA)!='N' && toupper(*_transA)!='T'
     && toupper(*_transA)!='C' && toupper(*_transA)!='R')
    {
        info = 2;
    } else if (toupper(*_diag)!='U' && toupper(*_diag)!='N') {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (ldA<std::max(1,n)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DTRMV ", &info);
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

//
//  Start the operations.
//
    if (lower) {
        if (!transA) {
            ulmBLAS::trlmv(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trumv(n, unitDiag, A, ldA, 1, x, incX);
        }
    } else {
        if (!transA) {
            ulmBLAS::trumv(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trlmv(n, unitDiag, A, ldA, 1, x, incX);
        }
    }
}

} // extern "C"
