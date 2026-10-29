#include <algorithm>
#include <cctype>
#include <complex>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(sgemv)(const char               *transA_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               const float              *A,
               const ULMBLAS_INDEX_TYPE *ldA_,
               const float              *x,
               const ULMBLAS_INDEX_TYPE *incX_,
               const float              *beta_,
               float                    *y,
               const ULMBLAS_INDEX_TYPE *incY_)
{
//
//  Dereference scalar parameters
//
    bool transA             = (toupper(*transA_) == 'T'
                            || toupper(*transA_) == 'C');
    ULMBLAS_INDEX_TYPE m    = *m_;
    ULMBLAS_INDEX_TYPE n    = *n_;
    float alpha             = *alpha_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    float beta              = *beta_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
     && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 1;
    } else if (m<0) {
        info = 2;
    } else if (n<0) {
            info = 3;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
            info = 6;
    } else if (incX==0) {
            info = 8;
    } else if (incY==0) {
            info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("SGEMV ", &info, 6);
        return;
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
        ulmBLAS::gemv<std::ptrdiff_t>(m, n, alpha, A, 1, ldA, x, incX,
                                      beta, y, incY);
    } else {
        ulmBLAS::gemv<std::ptrdiff_t>(n, m, alpha, A, ldA, 1, x, incX,
                                      beta, y, incY);
    }
}

void
F77BLAS(dgemv)(const char               *transA_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const double             *alpha_,
               const double             *A,
               const ULMBLAS_INDEX_TYPE *ldA_,
               const double             *x,
               const ULMBLAS_INDEX_TYPE *incX_,
               const double             *beta_,
               double                   *y,
               const ULMBLAS_INDEX_TYPE *incY_)
{
//
//  Dereference scalar parameters
//
    bool transA             = (toupper(*transA_) == 'T'
                            || toupper(*transA_) == 'C');
    ULMBLAS_INDEX_TYPE m    = *m_;
    ULMBLAS_INDEX_TYPE n    = *n_;
    double alpha            = *alpha_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    double beta             = *beta_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
     && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 1;
    } else if (m<0) {
        info = 2;
    } else if (n<0) {
            info = 3;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
            info = 6;
    } else if (incX==0) {
            info = 8;
    } else if (incY==0) {
            info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DGEMV ", &info, 6);
        return;
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
        ulmBLAS::gemv<std::ptrdiff_t>(m, n, alpha, A, 1, ldA, x, incX,
                                      beta, y, incY);
    } else {
        ulmBLAS::gemv<std::ptrdiff_t>(n, m, alpha, A, ldA, 1, x, incX,
                                      beta, y, incY);
    }
}

void
F77BLAS(cgemv)(const char               *transA_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               const float              *A_,
               const ULMBLAS_INDEX_TYPE *ldA_,
               const float              *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               const float              *beta_,
               float                    *y_,
               const ULMBLAS_INDEX_TYPE *incY_)
{
//
//  Dereference scalar parameters
//
    bool transA             = (toupper(*transA_) == 'T'
                            || toupper(*transA_) == 'C');
    bool conjA              = (toupper(*transA_) == 'R'
                            || toupper(*transA_) == 'C');
    ULMBLAS_INDEX_TYPE m    = *m_;
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

    typedef std::complex<float> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    dcomplex beta  = dcomplex(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);
//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
     && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 1;
    } else if (m<0) {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    } else if (incY==0) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CGEMV ", &info, 6);
        return;
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
        ulmBLAS::gemv<std::ptrdiff_t>(m, n, alpha, conjA, A, 1, ldA, x, incX,
                                      beta, y, incY);
    } else {
        ulmBLAS::gemv<std::ptrdiff_t>(n, m, alpha, conjA, A, ldA, 1, x, incX,
                                      beta, y, incY);
    }
}

void
F77BLAS(zgemv)(const char               *transA_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const double             *alpha_,
               const double             *A_,
               const ULMBLAS_INDEX_TYPE *ldA_,
               const double             *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               const double             *beta_,
               double                   *y_,
               const ULMBLAS_INDEX_TYPE *incY_)
{
//
//  Dereference scalar parameters
//
    bool transA             = (toupper(*transA_) == 'T'
                            || toupper(*transA_) == 'C');
    bool conjA              = (toupper(*transA_) == 'R'
                            || toupper(*transA_) == 'C');
    ULMBLAS_INDEX_TYPE m    = *m_;
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    dcomplex beta  = dcomplex(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);
//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
     && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 1;
    } else if (m<0) {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    } else if (incY==0) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZGEMV ", &info, 6);
        return;
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
        ulmBLAS::gemv<std::ptrdiff_t>(m, n, alpha, conjA, A, 1, ldA, x, incX,
                                      beta, y, incY);
    } else {
        ulmBLAS::gemv<std::ptrdiff_t>(n, m, alpha, conjA, A, ldA, 1, x, incX,
                                      beta, y, incY);
    }
}

} // extern "C"
