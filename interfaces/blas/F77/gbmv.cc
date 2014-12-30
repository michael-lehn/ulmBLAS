#include <algorithm>
#include <cctype>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/level2/gbmv.h>
#include <ulmblas/level2/gbmtv.h>

extern "C" {


void
F77BLAS(dgbmv)(const char     *transA_,
               const int      *m_,
               const int      *n_,
               const int      *kl_,
               const int      *ku_,
               const double   *alpha_,
               const double   *A,
               const int      *ldA_,
               const double   *x,
               const int      *incX_,
               const double   *beta_,
               double         *y,
               const int      *incY_)
{
//
//  Dereference scalar parameters
//
    bool transA  = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    int m        = *m_;
    int n        = *n_;
    int kl       = *kl_;
    int ku       = *ku_;
    double alpha = *alpha_;
    int ldA      = *ldA_;
    int incX     = *incX_;
    double beta  = *beta_;
    int incY     = *incY_;

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
     && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 1;
    } else if (m<0) {
        info = 2;
    } else if (n<0) {
            info = 3;
    } else if (kl<0) {
            info = 4;
    } else if (ku<0) {
            info = 5;
    } else if (ldA<kl+ku+1) {
            info = 8;
    } else if (incX==0) {
            info = 10;
    } else if (incY==0) {
            info = 13;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DGBMV ", &info);
    }

    if (!transA) {
        if (incX<0) {
            x -= incX*(n-1);
        }
        if (incY<0) {
            y -= incY*(m-1);
        }
    } else {
        if (incX<0) {
            x -= incX*(m-1);
        }
        if (incY<0) {
            y -= incY*(n-1);
        }
    }

//
//  Start the operations.
//
    if (!transA) {
        ulmBLAS::gbmv(m, n, kl, ku, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::gbmtv(m, n, kl, ku, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

} // extern "C"
