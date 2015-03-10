#include <algorithm>
#include <cctype>
#include <complex>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(sgemv)(const char     *transA_,
               const int      *m_,
               const int      *n_,
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
    } else if (ldA<std::max(1,m)) {
            info = 6;
    } else if (incX==0) {
            info = 8;
    } else if (incY==0) {
            info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("SGEMV ", &info);
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
        ulmBLAS::gemv(m, n, alpha, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::gemv(n, m, alpha, A, ldA, 1, x, incX, beta, y, incY);
    }
}

void
F77BLAS(dgemv)(const char     *transA_,
               const int      *m_,
               const int      *n_,
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
    } else if (ldA<std::max(1,m)) {
            info = 6;
    } else if (incX==0) {
            info = 8;
    } else if (incY==0) {
            info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DGEMV ", &info);
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
        ulmBLAS::gemv(m, n, alpha, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::gemv(n, m, alpha, A, ldA, 1, x, incX, beta, y, incY);
    }
}

void
F77BLAS(cgemv)(const char     *transA_,
               const int      *m_,
               const int      *n_,
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
    int ldA      = *ldA_;
    int incX     = *incX_;
    int incY     = *incY_;

    typedef std::complex<float> dcomplex;
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
    } else if (ldA<std::max(1,m)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    } else if (incY==0) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CGEMV ", &info);
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
        ulmBLAS::gemv(m, n, alpha, conjA, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::gemv(n, m, alpha, conjA, A, ldA, 1, x, incX, beta, y, incY);
    }
}

void
F77BLAS(zgemv)(const char     *transA_,
               const int      *m_,
               const int      *n_,
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
    } else if (ldA<std::max(1,m)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    } else if (incY==0) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZGEMV ", &info);
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
        ulmBLAS::gemv(m, n, alpha, conjA, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::gemv(n, m, alpha, conjA, A, ldA, 1, x, incX, beta, y, incY);
    }
}

} // extern "C"
