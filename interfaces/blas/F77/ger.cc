#include <algorithm>
#include <cctype>
#include <complex>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(sger)(const ULMBLAS_INDEX_TYPE *m_,
              const ULMBLAS_INDEX_TYPE *n_,
              const float              *alpha_,
              const float              *x,
              const ULMBLAS_INDEX_TYPE *incX_,
              const float              *y,
              const ULMBLAS_INDEX_TYPE *incY_,
              float                    *A,
              const ULMBLAS_INDEX_TYPE *ldA_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE m    = *m_;
    ULMBLAS_INDEX_TYPE n    = *n_;
    float alpha             = *alpha_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (m<0) {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (incX==0) {
        info = 5;
    } else if (incY==0) {
        info = 7;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("SGER  ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    ulmBLAS::ger<std::ptrdiff_t>(m, n, alpha, x, incX, y, incY, A, 1, ldA);
}


void
F77BLAS(dger)(const ULMBLAS_INDEX_TYPE *m_,
              const ULMBLAS_INDEX_TYPE *n_,
              const double             *alpha_,
              const double             *x,
              const ULMBLAS_INDEX_TYPE *incX_,
              const double             *y,
              const ULMBLAS_INDEX_TYPE *incY_,
              double                   *A,
              const ULMBLAS_INDEX_TYPE *ldA_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE m    = *m_;
    ULMBLAS_INDEX_TYPE n    = *n_;
    double alpha            = *alpha_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (m<0) {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (incX==0) {
        info = 5;
    } else if (incY==0) {
        info = 7;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DGER  ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    ulmBLAS::ger<std::ptrdiff_t>(m, n, alpha, x, incX, y, incY, A, 1, ldA);
}

void
F77BLAS(cgeru)(const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               const float              *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               const float              *y_,
               const ULMBLAS_INDEX_TYPE *incY_,
               float                    *A_,
               const ULMBLAS_INDEX_TYPE *ldA_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE m    = *m_;
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;

    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);

    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);
    fcomplex       *A = reinterpret_cast<fcomplex *>(A_);

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (m<0) {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (incX==0) {
        info = 5;
    } else if (incY==0) {
        info = 7;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CGERU ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    ulmBLAS::ger<std::ptrdiff_t>(m, n, alpha, x, incX, y, incY, A, 1, ldA);
}



void
F77BLAS(zgeru)(const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const double             *alpha_,
               const double             *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               const double             *y_,
               const ULMBLAS_INDEX_TYPE *incY_,
               double                   *A_,
               const ULMBLAS_INDEX_TYPE *ldA_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE m    = *m_;
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;

    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);

    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);
    dcomplex       *A = reinterpret_cast<dcomplex *>(A_);

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (m<0) {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (incX==0) {
        info = 5;
    } else if (incY==0) {
        info = 7;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZGERU ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    ulmBLAS::ger<std::ptrdiff_t>(m, n, alpha, x, incX, y, incY, A, 1, ldA);
}


void
F77BLAS(cgerc)(const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               const float              *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               const float              *y_,
               const ULMBLAS_INDEX_TYPE *incY_,
               float                    *A_,
               const ULMBLAS_INDEX_TYPE *ldA_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE m    = *m_;
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;

    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);

    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);
    fcomplex       *A = reinterpret_cast<fcomplex *>(A_);

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (m<0) {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (incX==0) {
        info = 5;
    } else if (incY==0) {
        info = 7;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CGERC ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    ulmBLAS::gerc<std::ptrdiff_t>(m, n, alpha, x, incX, y, incY, A, 1, ldA);
}



void
F77BLAS(zgerc)(const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const double             *alpha_,
               const double             *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               const double             *y_,
               const ULMBLAS_INDEX_TYPE *incY_,
               double                   *A_,
               const ULMBLAS_INDEX_TYPE *ldA_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE m    = *m_;
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;

    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);

    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);
    dcomplex       *A = reinterpret_cast<dcomplex *>(A_);

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (m<0) {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (incX==0) {
        info = 5;
    } else if (incY==0) {
        info = 7;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZGERC ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    ulmBLAS::gerc<std::ptrdiff_t>(m, n, alpha, x, incX, y, incY, A, 1, ldA);
}


} // extern "C"
