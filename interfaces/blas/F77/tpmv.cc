#include <algorithm>
#include <cctype>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/level2/tplmv.h>
#include <ulmblas/level2/tplmtv.h>
#include <ulmblas/level2/tpumv.h>
#include <ulmblas/level2/tpumtv.h>

extern "C" {

void
F77BLAS(dtpmv)(const char     *upLo_,
               const char     *transA_,
               const char     *diag_,
               const int      *n_,
               const double   *AP,
               double         *x,
               const int      *incX_)
{
//
//  Dereference scalar parameters
//
    bool lowerA   = (toupper(*upLo_) == 'L');
    bool transA   = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool unitDiag = (toupper(*diag_) == 'U');
    int n         = *n_;
    int incX      = *incX_;

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*upLo_)!='U' && toupper(*upLo_)!='L') {
        info = 1;
    } else if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
     && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 2;
    } else if (toupper(*diag_)!='U' && toupper(*diag_)!='N') {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (incX==0) {
        info = 7;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DTPMV ", &info);
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

//
//  Start the operations.
//
    if (!transA) {
        if (lowerA) {
            ulmBLAS::tplmv(n, unitDiag, AP, x, incX);
        } else {
            ulmBLAS::tpumv(n, unitDiag, AP, x, incX);
        }
    } else {
        if (lowerA) {
            ulmBLAS::tplmtv(n, unitDiag, AP, x, incX);
        } else {
            ulmBLAS::tpumtv(n, unitDiag, AP, x, incX);
        }
    }
}

} // extern "C"
