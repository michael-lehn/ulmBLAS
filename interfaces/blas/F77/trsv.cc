#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(strsv)(const char     *upLo_,
               const char     *transA_,
               const char     *diag_,
               const int      *n_,
               const float    *A,
               const int      *ldA_,
               float          *x,
               const int      *incX_)
{
//
//  Dereference scalar parameters
//
    bool lower    = (toupper(*upLo_) == 'L');
    bool transA   = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool unitDiag = (toupper(*diag_) == 'U');
    int  n        = *n_;
    int  ldA      = *ldA_;
    int  incX     = *incX_;

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
    } else if (ldA<std::max(1,n)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    }

    if (info!=0) {
        F77BLAS(xerbla)("STRSV ", &info);
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

//
//  Start the operations.
//
    if (lower) {
        if (!transA) {
            ulmBLAS::trlsv(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trusv(n, unitDiag, A, ldA, 1, x, incX);
        }
    } else {
        if (!transA) {
            ulmBLAS::trusv(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trlsv(n, unitDiag, A, ldA, 1, x, incX);
        }
    }
}


void
F77BLAS(dtrsv)(const char     *upLo_,
               const char     *transA_,
               const char     *diag_,
               const int      *n_,
               const double   *A,
               const int      *ldA_,
               double         *x,
               const int      *incX_)
{
//
//  Dereference scalar parameters
//
    bool lower    = (toupper(*upLo_) == 'L');
    bool transA   = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool unitDiag = (toupper(*diag_) == 'U');
    int  n        = *n_;
    int  ldA      = *ldA_;
    int  incX     = *incX_;

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
    } else if (ldA<std::max(1,n)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DTRSV ", &info);
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

//
//  Start the operations.
//
    if (lower) {
        if (!transA) {
            ulmBLAS::trlsv(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trusv(n, unitDiag, A, ldA, 1, x, incX);
        }
    } else {
        if (!transA) {
            ulmBLAS::trusv(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trlsv(n, unitDiag, A, ldA, 1, x, incX);
        }
    }
}

void
F77BLAS(ctrsv)(const char     *upLo_,
               const char     *transA_,
               const char     *diag_,
               const int      *n_,
               const float    *A_,
               const int      *ldA_,
               float          *x_,
               const int      *incX_)
{
//
//  Dereference scalar parameters
//
    bool lower    = (toupper(*upLo_) == 'L');
    bool transA   = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool conjA    = toupper(*transA_) == 'C';
    bool unitDiag = (toupper(*diag_) == 'U');
    int  n        = *n_;
    int  ldA      = *ldA_;
    int  incX     = *incX_;

    typedef std::complex<float> fcomplex;
    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    fcomplex       *x = reinterpret_cast<fcomplex *>(x_);

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
    } else if (ldA<std::max(1,n)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CTRSV ", &info);
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

//
//  Start the operations.
//
    if (lower) {
        if (!transA) {
            ulmBLAS::trlsv(n, unitDiag, conjA, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trusv(n, unitDiag, conjA, A, ldA, 1, x, incX);
        }
    } else {
        if (!transA) {
            ulmBLAS::trusv(n, unitDiag, conjA, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trlsv(n, unitDiag, conjA, A, ldA, 1, x, incX);
        }
    }
}

void
F77BLAS(ztrsv)(const char     *upLo_,
               const char     *transA_,
               const char     *diag_,
               const int      *n_,
               const double   *A_,
               const int      *ldA_,
               double         *x_,
               const int      *incX_)
{
//
//  Dereference scalar parameters
//
    bool lower    = (toupper(*upLo_) == 'L');
    bool transA   = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool conjA    = toupper(*transA_) == 'C';
    bool unitDiag = (toupper(*diag_) == 'U');
    int  n        = *n_;
    int  ldA      = *ldA_;
    int  incX     = *incX_;

    typedef std::complex<double> dcomplex;
    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    dcomplex       *x = reinterpret_cast<dcomplex *>(x_);

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
    } else if (ldA<std::max(1,n)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZTRSV ", &info);
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

//
//  Start the operations.
//
    if (lower) {
        if (!transA) {
            ulmBLAS::trlsv(n, unitDiag, conjA, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trusv(n, unitDiag, conjA, A, ldA, 1, x, incX);
        }
    } else {
        if (!transA) {
            ulmBLAS::trusv(n, unitDiag, conjA, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trlsv(n, unitDiag, conjA, A, ldA, 1, x, incX);
        }
    }
}


} // extern "C"
