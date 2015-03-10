#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

//#define SCATTER

#ifdef SCATTER
#define   SCATTER_INCROWA   1
#define   SCATTER_INCCOLA   1
#define   SCATTER_INCX      2
#define   SCATTER_INCY      1
#endif


extern "C" {

void
ULMBLAS(sger)(int           m,
              int           n,
              float         alpha,
              const float   *x,
              int           incX,
              const float   *y,
              int           incY,
              float         *A,
              int           ldA)
{
    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

//
//  Start the operations.
//
    ulmBLAS::ger(m, n, alpha, x, incX, y, incY, A, 1, ldA);
}

void
ULMBLAS(dger)(int           m,
              int           n,
              double        alpha,
              const double  *x,
              int           incX,
              const double  *y,
              int           incY,
              double        *A,
              int           ldA)
{
    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

#ifndef SCATTER
//
//  Start the operations.
//
    ulmBLAS::ger(m, n, alpha, x, incX, y, incY, A, 1, ldA);

#else
//
//  Scatter operands
//
    double *x_ = new double[m*incX*SCATTER_INCX];
    double *y_ = new double[n*incY*SCATTER_INCY];
    double *A_ = new double[ldA*n*SCATTER_INCROWA*SCATTER_INCCOLA];

    ulmBLAS::copy(m, false, x, incX, x_, incX*SCATTER_INCX);
    ulmBLAS::copy(m, false, y, incY, y_, incY*SCATTER_INCY);
    ulmBLAS::gecopy(m, n, false,
                    A, 1, ldA,
                    A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA);

//
//  Start the operations.
//
    ulmBLAS::ger(m, n, alpha,
                 x_, incX*SCATTER_INCX,
                 y_, incY*SCATTER_INCY,
                 A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA);

    ulmBLAS::gecopy(m, n, A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA, A, 1, ldA);
//
//  Gather result
//
    delete [] x_;
    delete [] y_;
    delete [] A_;
#endif

}

void
ULMBLAS(cgeru)(int           m,
               int           n,
               const float   *alpha_,
               const float   *x_,
               int           incX,
               const float   *y_,
               int           incY,
               float         *A_,
               int           ldA)
{
//
//  Start the operations.
//
    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);

    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);
    fcomplex       *A = reinterpret_cast<fcomplex *>(A_);

    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    ulmBLAS::ger(m, n, alpha, x, incX, y, incY, A, 1, ldA);
}

void
ULMBLAS(zgeru)(int           m,
               int           n,
               const double  *alpha_,
               const double  *x_,
               int           incX,
               const double  *y_,
               int           incY,
               double        *A_,
               int           ldA)
{
//
//  Start the operations.
//
    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);

    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);
    dcomplex       *A = reinterpret_cast<dcomplex *>(A_);

    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    ulmBLAS::ger(m, n, alpha, x, incX, y, incY, A, 1, ldA);
}

void
ULMBLAS(cgerc)(int           m,
               int           n,
               const float   *alpha_,
               const float   *x_,
               int           incX,
               const float   *y_,
               int           incY,
               float         *A_,
               int           ldA)
{
//
//  Start the operations.
//
    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);

    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);
    fcomplex       *A = reinterpret_cast<fcomplex *>(A_);

    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    ulmBLAS::gerc(m, n, alpha, x, incX, y, incY, A, 1, ldA);
}

void
ULMBLAS(cgerc_)(int           m,
                int           n,
                const float   *alpha_,
                const float   *x_,
                int           incX,
                const float   *y_,
                int           incY,
                float         *A_,
                int           ldA)
{
//
//  Start the operations.
//
    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);

    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);
    fcomplex       *A = reinterpret_cast<fcomplex *>(A_);

    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    ulmBLAS::gerc(m, n, alpha, true, x, incX, y, incY, A, 1, ldA);
}


void
ULMBLAS(zgerc)(int           m,
               int           n,
               const double  *alpha_,
               const double  *x_,
               int           incX,
               const double  *y_,
               int           incY,
               double        *A_,
               int           ldA)
{
//
//  Start the operations.
//
    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);

    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);
    dcomplex       *A = reinterpret_cast<dcomplex *>(A_);

    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    ulmBLAS::gerc(m, n, alpha, x, incX, y, incY, A, 1, ldA);
}

void
ULMBLAS(zgerc_)(int           m,
                int           n,
                const double  *alpha_,
                const double  *x_,
                int           incX,
                const double  *y_,
                int           incY,
                double        *A_,
                int           ldA)
{
//
//  Start the operations.
//
    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);

    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);
    dcomplex       *A = reinterpret_cast<dcomplex *>(A_);

    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    ulmBLAS::gerc(m, n, alpha, true, x, incX, y, incY, A, 1, ldA);
}

void
CBLAS(sger)(enum CBLAS_ORDER  order,
            int               m,
            int               n,
            float             alpha,
            const float       *x,
            int               incX,
            const float       *y,
            int               incY,
            float             *A,
            int               ldA)
{
//
//  Test the input parameters
//
    int ldAmin = (order==CblasColMajor) ? std::max(1,m) : std::max(1,n);
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 2;
            } else if (n<0) {
                info = 3;
            }
        } else {
            if (n<0) {
                info = 2;
            } else if (m<0) {
                info = 3;
            }
        }
    }
    if (info==0) {
        if (order==CblasColMajor) {
            if (incX==0) {
                info = 6;
            } else if (incY==0) {
                info = 8;
            }
        } else {
            if (incY==0) {
                info = 6;
            } else if (incX==0) {
                info = 8;
            }
        }
    }
    if (info==0) {
        if (ldA<ldAmin) {
            info = 10;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_sger", "... bla bla ...");
    }


    if (order==CblasColMajor) {
        ULMBLAS(sger)(m, n, alpha, x, incX, y, incY, A, ldA);
    } else{
        ULMBLAS(sger)(n, m, alpha, y, incY, x, incX, A, ldA);
    }

}

void
CBLAS(dger)(enum CBLAS_ORDER  order,
            int               m,
            int               n,
            double            alpha,
            const double      *x,
            int               incX,
            const double      *y,
            int               incY,
            double            *A,
            int               ldA)
{
//
//  Test the input parameters
//
    int ldAmin = (order==CblasColMajor) ? std::max(1,m) : std::max(1,n);
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 2;
            } else if (n<0) {
                info = 3;
            }
        } else {
            if (n<0) {
                info = 2;
            } else if (m<0) {
                info = 3;
            }
        }
    }
    if (info==0) {
        if (order==CblasColMajor) {
            if (incX==0) {
                info = 6;
            } else if (incY==0) {
                info = 8;
            }
        } else {
            if (incY==0) {
                info = 6;
            } else if (incX==0) {
                info = 8;
            }
        }
    }
    if (info==0) {
        if (ldA<ldAmin) {
            info = 10;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dger", "... bla bla ...");
    }


    if (order==CblasColMajor) {
        ULMBLAS(dger)(m, n, alpha, x, incX, y, incY, A, ldA);
    } else{
        ULMBLAS(dger)(n, m, alpha, y, incY, x, incX, A, ldA);
    }

}

void
CBLAS(cgeru)(enum CBLAS_ORDER  order,
             int               m,
             int               n,
             const float       *alpha,
             const float       *x,
             int               incX,
             const float       *y,
             int               incY,
             float             *A,
             int               ldA)
{
//
//  Test the input parameters
//
    int ldAmin = (order==CblasColMajor) ? std::max(1,m) : std::max(1,n);
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 2;
            } else if (n<0) {
                info = 3;
            }
        } else {
            if (n<0) {
                info = 2;
            } else if (m<0) {
                info = 3;
            }
        }
    }
    if (info==0) {
        if (order==CblasColMajor) {
            if (incX==0) {
                info = 6;
            } else if (incY==0) {
                info = 8;
            }
        } else {
            if (incY==0) {
                info = 6;
            } else if (incX==0) {
                info = 8;
            }
        }
    }
    if (info==0) {
        if (ldA<ldAmin) {
            info = 10;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_cgeru", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(cgeru)(m, n, alpha, x, incX, y, incY, A, ldA);
    } else {
        ULMBLAS(cgeru)(n, m, alpha, y, incY, x, incX, A, ldA);
    }

}

void
CBLAS(zgeru)(enum CBLAS_ORDER  order,
             int               m,
             int               n,
             const double      *alpha,
             const double      *x,
             int               incX,
             const double      *y,
             int               incY,
             double            *A,
             int               ldA)
{
//
//  Test the input parameters
//
    int ldAmin = (order==CblasColMajor) ? std::max(1,m) : std::max(1,n);
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 2;
            } else if (n<0) {
                info = 3;
            }
        } else {
            if (n<0) {
                info = 2;
            } else if (m<0) {
                info = 3;
            }
        }
    }
    if (info==0) {
        if (order==CblasColMajor) {
            if (incX==0) {
                info = 6;
            } else if (incY==0) {
                info = 8;
            }
        } else {
            if (incY==0) {
                info = 6;
            } else if (incX==0) {
                info = 8;
            }
        }
    }
    if (info==0) {
        if (ldA<ldAmin) {
            info = 10;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_zgeru", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(zgeru)(m, n, alpha, x, incX, y, incY, A, ldA);
    } else {
        ULMBLAS(zgeru)(n, m, alpha, y, incY, x, incX, A, ldA);
    }

}

void
CBLAS(cgerc)(enum CBLAS_ORDER  order,
             int               m,
             int               n,
             const float       *alpha,
             const float       *x,
             int               incX,
             const float       *y,
             int               incY,
             float             *A,
             int               ldA)
{
//
//  Test the input parameters
//
    int ldAmin = (order==CblasColMajor) ? std::max(1,m) : std::max(1,n);
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 2;
            } else if (n<0) {
                info = 3;
            }
        } else {
            if (n<0) {
                info = 2;
            } else if (m<0) {
                info = 3;
            }
        }
    }
    if (info==0) {
        if (order==CblasColMajor) {
            if (incX==0) {
                info = 6;
            } else if (incY==0) {
                info = 8;
            }
        } else {
            if (incY==0) {
                info = 6;
            } else if (incX==0) {
                info = 8;
            }
        }
    }
    if (info==0) {
        if (ldA<ldAmin) {
            info = 10;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_cgerc", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(cgerc)(m, n, alpha, x, incX, y, incY, A, ldA);
    } else {
        ULMBLAS(cgerc_)(n, m, alpha, y, incY, x, incX, A, ldA);
    }

}

void
CBLAS(zgerc)(enum CBLAS_ORDER  order,
             int               m,
             int               n,
             const double      *alpha,
             const double      *x,
             int               incX,
             const double      *y,
             int               incY,
             double            *A,
             int               ldA)
{
//
//  Test the input parameters
//
    int ldAmin = (order==CblasColMajor) ? std::max(1,m) : std::max(1,n);
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 2;
            } else if (n<0) {
                info = 3;
            }
        } else {
            if (n<0) {
                info = 2;
            } else if (m<0) {
                info = 3;
            }
        }
    }
    if (info==0) {
        if (order==CblasColMajor) {
            if (incX==0) {
                info = 6;
            } else if (incY==0) {
                info = 8;
            }
        } else {
            if (incY==0) {
                info = 6;
            } else if (incX==0) {
                info = 8;
            }
        }
    }
    if (info==0) {
        if (ldA<ldAmin) {
            info = 10;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_zgerc", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(zgerc)(m, n, alpha, x, incX, y, incY, A, ldA);
    } else {
        ULMBLAS(zgerc_)(n, m, alpha, y, incY, x, incX, A, ldA);
    }

}

} // extern "C"
