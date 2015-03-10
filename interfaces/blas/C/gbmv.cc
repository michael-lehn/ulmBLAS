#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(sgbmv)(enum CBLAS_TRANSPOSE  transA,
               int                   m,
               int                   n,
               int                   kl,
               int                   ku,
               float                 alpha,
               const float           *A,
               int                   ldA,
               const float           *x,
               int                   incX,
               float                 beta,
               float                 *y,
               int                   incY)
{
    if (transA==CblasNoTrans || transA==AtlasConj) {
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
    if (transA==CblasNoTrans || transA==AtlasConj) {
        ulmBLAS::gbmv(m, n, kl, ku, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::gbmtv(m, n, kl, ku, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
ULMBLAS(dgbmv)(enum CBLAS_TRANSPOSE  transA,
               int                   m,
               int                   n,
               int                   kl,
               int                   ku,
               double                alpha,
               const double          *A,
               int                   ldA,
               const double          *x,
               int                   incX,
               double                beta,
               double                *y,
               int                   incY)
{
    if (transA==CblasNoTrans || transA==AtlasConj) {
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
    if (transA==CblasNoTrans || transA==AtlasConj) {
        ulmBLAS::gbmv(m, n, kl, ku, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::gbmtv(m, n, kl, ku, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
ULMBLAS(cgbmv)(enum CBLAS_TRANSPOSE  transA_,
               int                   m,
               int                   n,
               int                   kl,
               int                   ku,
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
ULMBLAS(zgbmv)(enum CBLAS_TRANSPOSE  transA_,
               int                   m,
               int                   n,
               int                   kl,
               int                   ku,
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
CBLAS(sgbmv)(enum CBLAS_ORDER      order,
             enum CBLAS_TRANSPOSE  transA,
             int                   m,
             int                   n,
             int                   kl,
             int                   ku,
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
    }
    if (info==0) {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 3;
            } else if (n<0) {
                    info = 4;
            } else if (kl<0) {
                    info = 5;
            } else if (ku<0) {
                    info = 6;
            } else if (ldA<kl+ku+1) {
                    info = 9;
            }
        } else {
            if (n<0) {
                info = 3;
            } else if (m<0) {
                info = 4;
            } else if (ku<0) {
                info = 5;
            } else if (kl<0) {
                info = 6;
            } else if (ldA<kl+ku+1) {
                info = 9;
            }
        }
    }
    if (info==0) {
        if (incX==0) {
            info = 11;
        } else if (incY==0) {
            info = 14;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_sgbmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(sgbmv)(transA, m, n, kl, ku,
                       alpha,
                       A, ldA,
                       x, incX,
                       beta,
                       y, incY);
    } else {
        transA = transpose(transA);
        ULMBLAS(sgbmv)(transA, n, m, ku, kl,
                       alpha,
                       A, ldA,
                       x, incX,
                       beta,
                       y, incY);
     }
}


void
CBLAS(dgbmv)(enum CBLAS_ORDER      order,
             enum CBLAS_TRANSPOSE  transA,
             int                   m,
             int                   n,
             int                   kl,
             int                   ku,
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
    }
    if (info==0) {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 3;
            } else if (n<0) {
                    info = 4;
            } else if (kl<0) {
                    info = 5;
            } else if (ku<0) {
                    info = 6;
            } else if (ldA<kl+ku+1) {
                    info = 9;
            }
        } else {
            if (n<0) {
                info = 3;
            } else if (m<0) {
                info = 4;
            } else if (ku<0) {
                info = 5;
            } else if (kl<0) {
                info = 6;
            } else if (ldA<kl+ku+1) {
                info = 9;
            }
        }
    }
    if (info==0) {
        if (incX==0) {
            info = 11;
        } else if (incY==0) {
            info = 14;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dgbmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dgbmv)(transA, m, n, kl, ku,
                       alpha,
                       A, ldA,
                       x, incX,
                       beta,
                       y, incY);
    } else {
        transA = transpose(transA);
        ULMBLAS(dgbmv)(transA, n, m, ku, kl,
                       alpha,
                       A, ldA,
                       x, incX,
                       beta,
                       y, incY);
     }
}

void
CBLAS(cgbmv)(enum CBLAS_ORDER      order,
             enum CBLAS_TRANSPOSE  transA,
             int                   m,
             int                   n,
             int                   kl,
             int                   ku,
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
    }
    if (info==0) {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 3;
            } else if (n<0) {
                    info = 4;
            } else if (kl<0) {
                    info = 5;
            } else if (ku<0) {
                    info = 6;
            } else if (ldA<kl+ku+1) {
                    info = 9;
            }
        } else {
            if (n<0) {
                info = 3;
            } else if (m<0) {
                info = 4;
            } else if (ku<0) {
                info = 5;
            } else if (kl<0) {
                info = 6;
            } else if (ldA<kl+ku+1) {
                info = 9;
            }
        }
    }
    if (info==0) {
        if (incX==0) {
            info = 11;
        } else if (incY==0) {
            info = 14;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_cgbmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(cgbmv)(transA, m, n, kl, ku,
                       alpha,
                       A, ldA,
                       x, incX,
                       beta,
                       y, incY);
    } else {
        transA = transpose(transA);
        ULMBLAS(cgbmv)(transA, n, m, ku, kl,
                       alpha,
                       A, ldA,
                       x, incX,
                       beta,
                       y, incY);
     }
}
void
CBLAS(zgbmv)(enum CBLAS_ORDER      order,
             enum CBLAS_TRANSPOSE  transA,
             int                   m,
             int                   n,
             int                   kl,
             int                   ku,
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
    }
    if (info==0) {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 3;
            } else if (n<0) {
                    info = 4;
            } else if (kl<0) {
                    info = 5;
            } else if (ku<0) {
                    info = 6;
            } else if (ldA<kl+ku+1) {
                    info = 9;
            }
        } else {
            if (n<0) {
                info = 3;
            } else if (m<0) {
                info = 4;
            } else if (ku<0) {
                info = 5;
            } else if (kl<0) {
                info = 6;
            } else if (ldA<kl+ku+1) {
                info = 9;
            }
        }
    }
    if (info==0) {
        if (incX==0) {
            info = 11;
        } else if (incY==0) {
            info = 14;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_zgbmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(zgbmv)(transA, m, n, kl, ku,
                       alpha,
                       A, ldA,
                       x, incX,
                       beta,
                       y, incY);
    } else {
        transA = transpose(transA);
        ULMBLAS(zgbmv)(transA, n, m, ku, kl,
                       alpha,
                       A, ldA,
                       x, incX,
                       beta,
                       y, incY);
     }
}

} // extern "C"
