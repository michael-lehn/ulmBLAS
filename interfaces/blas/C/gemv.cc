#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

//#define SCATTER

#ifdef SCATTER
#define   SCATTER_INCROWA   2
#define   SCATTER_INCCOLA   3
#define   SCATTER_INCX      4
#define   SCATTER_INCY      5
#endif


extern "C" {

void
ULMBLAS(sgemv)(enum CBLAS_TRANSPOSE  transA,
               int                   m,
               int                   n,
               float                 alpha,
               const float           *A,
               int                   ldA,
               const float           *x,
               int                   incX,
               float                 beta,
               float                 *y,
               int                   incY)
{
//
//  Start the operations.
//
    if (transA==CblasNoTrans || transA==AtlasConj) {
        if (incX<0) {
            x -= incX*(n-1);
        }
        if (incY<0) {
            y -= incY*(m-1);
        }
        ulmBLAS::gemv(m, n, alpha, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        if (incX<0) {
            x -= incX*(m-1);
        }
        if (incY<0) {
            y -= incY*(n-1);
        }
        ulmBLAS::gemv(n, m, alpha, A, ldA, 1, x, incX, beta, y, incY);
    }
}

void
ULMBLAS(dgemv)(enum CBLAS_TRANSPOSE  transA,
               int                   m,
               int                   n,
               double                alpha,
               const double          *A,
               int                   ldA,
               const double          *x,
               int                   incX,
               double                beta,
               double                *y,
               int                   incY)
{
#ifndef SCATTER
//
//  Start the operations.
//
    if (transA==CblasNoTrans || transA==AtlasConj) {
        if (incX<0) {
            x -= incX*(n-1);
        }
        if (incY<0) {
            y -= incY*(m-1);
        }
        ulmBLAS::gemv(m, n, alpha, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        if (incX<0) {
            x -= incX*(m-1);
        }
        if (incY<0) {
            y -= incY*(n-1);
        }
        ulmBLAS::gemv(n, m, alpha, A, ldA, 1, x, incX, beta, y, incY);
    }
#else
    if (transA==CblasNoTrans || transA==AtlasConj) {
//
//      Scatter operands
//
        double *A_ = new double[ldA*n*SCATTER_INCROWA*SCATTER_INCCOLA];
        double *x_ = new double[n*incX*SCATTER_INCX];
        double *y_ = new double[m*incY*SCATTER_INCY];

        ulmBLAS::gecopy(m, n,
                        A, 1, ldA,
                        A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA);
        ulmBLAS::copy(n, x, incX, x_, incX*SCATTER_INCX);
        ulmBLAS::copy(m, y, incY, y_, incY*SCATTER_INCY);

//
//      Start the operations.
//
        ulmBLAS::gemv(m, n, alpha,
                      A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA,
                      x_, incX*SCATTER_INCX,
                      beta,
                      y_, incY*SCATTER_INCY);
        ulmBLAS::copy(m, y_, incY*SCATTER_INCY, y, incY);

//
//      Gather result
//
        delete [] A_;
        delete [] x_;
        delete [] y_;
    } else {
//
//      Scatter operands
//
        double *A_ = new double[ldA*n*SCATTER_INCROWA*SCATTER_INCCOLA];
        double *x_ = new double[m*incX*SCATTER_INCX];
        double *y_ = new double[n*incY*SCATTER_INCY];

        ulmBLAS::gecopy(m, n,
                        A, 1, ldA,
                        A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA);
        ulmBLAS::copy(n, x, incX, x_, incX*SCATTER_INCX);
        ulmBLAS::copy(m, y, incY, y_, incY*SCATTER_INCY);

//
//      Start the operations.
//
        ulmBLAS::gemv(n, m, alpha,
                      A_, ldA*SCATTER_INCCOLA, SCATTER_INCROWA,
                      x_, incX*SCATTER_INCX,
                      beta,
                      y_, incY*SCATTER_INCY);
        ulmBLAS::copy(m, y_, incY*SCATTER_INCY, y, incY);

//
//      Gather result
//
        delete [] A_;
        delete [] x_;
        delete [] y_;
    }
#endif
}

void
ULMBLAS(cgemv)(enum CBLAS_TRANSPOSE  transA_,
               int                   m,
               int                   n,
               const float           *alpha_,
               const float           *A_,
               int                   ldA,
               const float           *x_,
               int                   incX,
               const float           *beta_,
               float                 *y_,
               int                   incY)
{
    bool transA = (transA_==CblasTrans || transA_==CblasConjTrans);
    bool conjA  = (transA_==AtlasConj || transA_==CblasConjTrans);

    typedef std::complex<float> fcomplex;
    fcomplex       alpha = fcomplex(alpha_[0], alpha_[1]);
    fcomplex       beta  = fcomplex(beta_[0], beta_[1]);
    const fcomplex *A    = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *x    = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *y    = reinterpret_cast<fcomplex *>(y_);

//
//  Start the operations.
//
    if (!transA) {
        if (incX<0) {
            x -= incX*(n-1);
        }
        if (incY<0) {
            y -= incY*(m-1);
        }
        ulmBLAS::gemv(m, n, alpha, conjA, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        if (incX<0) {
            x -= incX*(m-1);
        }
        if (incY<0) {
            y -= incY*(n-1);
        }
        ulmBLAS::gemv(n, m, alpha, conjA, A, ldA, 1, x, incX, beta, y, incY);
    }
}

void
ULMBLAS(zgemv)(enum CBLAS_TRANSPOSE  transA_,
               int                   m,
               int                   n,
               const double          *alpha_,
               const double          *A_,
               int                   ldA,
               const double          *x_,
               int                   incX,
               const double          *beta_,
               double                *y_,
               int                   incY)
{
    bool transA = (transA_==CblasTrans || transA_==CblasConjTrans);
    bool conjA  = (transA_==AtlasConj || transA_==CblasConjTrans);

    typedef std::complex<double> dcomplex;
    dcomplex       alpha = dcomplex(alpha_[0], alpha_[1]);
    dcomplex       beta  = dcomplex(beta_[0], beta_[1]);
    const dcomplex *A    = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *x    = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y    = reinterpret_cast<dcomplex *>(y_);

//
//  Start the operations.
//
    if (!transA) {
        if (incX<0) {
            x -= incX*(n-1);
        }
        if (incY<0) {
            y -= incY*(m-1);
        }
        ulmBLAS::gemv(m, n, alpha, conjA, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        if (incX<0) {
            x -= incX*(m-1);
        }
        if (incY<0) {
            y -= incY*(n-1);
        }
        ulmBLAS::gemv(n, m, alpha, conjA, A, ldA, 1, x, incX, beta, y, incY);
    }
}

void
CBLAS(sgemv)(enum CBLAS_ORDER      order,
             enum CBLAS_TRANSPOSE  transA,
             int                   m,
             int                   n,
             float                 alpha,
             const float           *A,
             int                   ldA,
             const float           *x,
             int                   incX,
             float                 beta,
             float                 *y,
             int                   incY)
{
//
//  Test the input parameters
//
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (transA!=CblasNoTrans && transA!=AtlasConj
            && transA!=CblasTrans && transA!=CblasConjTrans)
    {
        info = 2;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 3;
            } else if (n<0) {
                info = 4;
            } else if (ldA<std::max(1,m)) {
                info = 7;
            }
        } else {
            if (n<0) {
                info = 3;
            } else if (m<0) {
                info = 4;
            } else if (ldA<std::max(1,n)) {
                info = 7;
            }
        }
    }
    if (info==0) {
        if (incX==0) {
            info = 9;
        } else if (incY==0) {
            info = 12;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_sgemv", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(sgemv)(transA, m, n, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        transA = transpose(transA);
        ULMBLAS(sgemv)(transA, n, m, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
CBLAS(dgemv)(enum CBLAS_ORDER      order,
             enum CBLAS_TRANSPOSE  transA,
             int                   m,
             int                   n,
             double                alpha,
             const double          *A,
             int                   ldA,
             const double          *x,
             int                   incX,
             double                beta,
             double                *y,
             int                   incY)
{
//
//  Test the input parameters
//
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (transA!=CblasNoTrans && transA!=AtlasConj
            && transA!=CblasTrans && transA!=CblasConjTrans)
    {
        info = 2;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 3;
            } else if (n<0) {
                info = 4;
            } else if (ldA<std::max(1,m)) {
                info = 7;
            }
        } else {
            if (n<0) {
                info = 3;
            } else if (m<0) {
                info = 4;
            } else if (ldA<std::max(1,n)) {
                info = 7;
            }
        }
    }
    if (info==0) {
        if (incX==0) {
            info = 9;
        } else if (incY==0) {
            info = 12;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dgemv", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(dgemv)(transA, m, n, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        transA = transpose(transA);
        ULMBLAS(dgemv)(transA, n, m, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
CBLAS(cgemv)(enum CBLAS_ORDER      order,
             enum CBLAS_TRANSPOSE  transA,
             int                   m,
             int                   n,
             const float           *alpha,
             const float           *A,
             int                   ldA,
             const float           *x,
             int                   incX,
             const float           *beta,
             float                 *y,
             int                   incY)
{
//
//  Test the input parameters
//
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (transA!=CblasNoTrans && transA!=AtlasConj
            && transA!=CblasTrans && transA!=CblasConjTrans)
    {
        info = 2;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 3;
            } else if (n<0) {
                info = 4;
            } else if (ldA<std::max(1,m)) {
                info = 7;
            }
        } else {
            if (n<0) {
                info = 3;
            } else if (m<0) {
                info = 4;
            } else if (ldA<std::max(1,n)) {
                info = 7;
            }
        }
    }
    if (info==0) {
        if (incX==0) {
            info = 9;
        } else if (incY==0) {
            info = 12;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_cgemv", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(cgemv)(transA, m, n, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        transA = transpose(transA);
        ULMBLAS(cgemv)(transA, n, m, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
CBLAS(zgemv)(enum CBLAS_ORDER      order,
             enum CBLAS_TRANSPOSE  transA,
             int                   m,
             int                   n,
             const double          *alpha,
             const double          *A,
             int                   ldA,
             const double          *x,
             int                   incX,
             const double          *beta,
             double                *y,
             int                   incY)
{
//
//  Test the input parameters
//
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (transA!=CblasNoTrans && transA!=AtlasConj
            && transA!=CblasTrans && transA!=CblasConjTrans)
    {
        info = 2;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 3;
            } else if (n<0) {
                info = 4;
            } else if (ldA<std::max(1,m)) {
                info = 7;
            }
        } else {
            if (n<0) {
                info = 3;
            } else if (m<0) {
                info = 4;
            } else if (ldA<std::max(1,n)) {
                info = 7;
            }
        }
    }
    if (info==0) {
        if (incX==0) {
            info = 9;
        } else if (incY==0) {
            info = 12;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_zgemv", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(zgemv)(transA, m, n, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        transA = transpose(transA);
        ULMBLAS(zgemv)(transA, n, m, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

} // extern "C"
