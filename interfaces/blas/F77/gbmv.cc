#include <algorithm>
#include <cctype>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(sgbmv)(const char     *transA_,
               const int      *m_,
               const int      *n_,
               const int      *kl_,
               const int      *ku_,
               const float    *alpha_,
               const float    *A,
               const int      *ldA_,
               const float    *x,
               const int      *incX_,
               const float    *beta_,
               float          *y,
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
    float alpha  = *alpha_;
    int ldA      = *ldA_;
    int incX     = *incX_;
    float beta   = *beta_;
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
        F77BLAS(xerbla)("SGBMV ", &info);
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

void
F77BLAS(cgbmv)(const char     *transA_,
               const int      *m_,
               const int      *n_,
               const int      *kl_,
               const int      *ku_,
               const float    *alpha_,
               const float    *A_,
               const int      *ldA_,
               const float    *x_,
               const int      *incX_,
               const float    *beta_,
               float          *y_,
               const int      *incY_)
{
//
//  Dereference scalar parameters
//
    bool transA  = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool conjA   = (toupper(*transA_) == 'R' || toupper(*transA_) == 'C');
    int m        = *m_;
    int n        = *n_;
    int kl       = *kl_;
    int ku       = *ku_;
    int ldA      = *ldA_;
    int incX     = *incX_;
    int incY     = *incY_;

    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);
    fcomplex beta  = fcomplex(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *y = reinterpret_cast<fcomplex *>(y_);

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
        F77BLAS(xerbla)("CGBMV ", &info);
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
        ulmBLAS::gbmv(m, n, kl, ku, alpha,
                      conjA, A, ldA,
                      x, incX,
                      beta,
                      y, incY);
    } else {
        ulmBLAS::gbmtv(m, n, kl, ku, alpha,
                       conjA, A, ldA,
                       x, incX,
                       beta,
                       y, incY);
    }
}


void
F77BLAS(zgbmv)(const char     *transA_,
               const int      *m_,
               const int      *n_,
               const int      *kl_,
               const int      *ku_,
               const double   *alpha_,
               const double   *A_,
               const int      *ldA_,
               const double   *x_,
               const int      *incX_,
               const double   *beta_,
               double         *y_,
               const int      *incY_)
{
//
//  Dereference scalar parameters
//
    bool transA  = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool conjA   = (toupper(*transA_) == 'R' || toupper(*transA_) == 'C');
    int m        = *m_;
    int n        = *n_;
    int kl       = *kl_;
    int ku       = *ku_;
    int ldA      = *ldA_;
    int incX     = *incX_;
    int incY     = *incY_;

    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    dcomplex beta  = dcomplex(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);

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
        F77BLAS(xerbla)("ZGBMV ", &info);
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
        ulmBLAS::gbmv(m, n, kl, ku, alpha,
                      conjA, A, ldA,
                      x, incX,
                      beta,
                      y, incY);
    } else {
        ulmBLAS::gbmtv(m, n, kl, ku, alpha,
                       conjA, A, ldA,
                       x, incX,
                       beta,
                       y, incY);
    }
}

} // extern "C"
