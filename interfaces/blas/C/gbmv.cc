#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level2/gbmv.h>
#include <ulmblas/level2/gbmtv.h>

extern "C" {

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

} // extern "C"
