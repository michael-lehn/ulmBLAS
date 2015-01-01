#include <algorithm>
#include <cctype>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/level2/tblmv.h>
#include <ulmblas/level2/tbumv.h>
#include <ulmblas/level2/tblmtv.h>
#include <ulmblas/level2/tbumtv.h>

extern "C" {

void
F77BLAS(dtbmv)(const char     *upLo_,
               const char     *transA_,
               const char     *diag_,
               const int      *n_,
               const int      *k_,
               const double   *A,
               const int      *ldA_,
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
    int k         = *k_;
    int ldA       = *ldA_;
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
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max(1,k+1)) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DTBMV ", &info);
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

//
//  Start the operations.
//
    if (!transA) {
        if (lowerA) {
            ulmBLAS::tblmv(n, k, unitDiag, A, ldA, x, incX);
        } else {
            ulmBLAS::tbumv(n, k, unitDiag, A, ldA, x, incX);
        }
    } else {
        if (lowerA) {
            ulmBLAS::tblmtv(n, k, unitDiag, A, ldA, x, incX);
        } else {
            ulmBLAS::tbumtv(n, k, unitDiag, A, ldA, x, incX);
        }
    }
}

} // extern "C"
