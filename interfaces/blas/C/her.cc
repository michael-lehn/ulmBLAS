
#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(cher)(enum CBLAS_UPLO    upLo,
              ULMBLAS_INDEX_TYPE n,
              float              alpha,
              const float        *x_,
              ULMBLAS_INDEX_TYPE incX,
              float              *A_,
              ULMBLAS_INDEX_TYPE ldA)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *A = reinterpret_cast<fcomplex *>(A_);
//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (upLo==CblasLower) {
        ulmBLAS::helr<std::ptrdiff_t>(n, alpha, false, x, incX, A, 1, ldA);
    } else {
        ulmBLAS::helr<std::ptrdiff_t>(n, alpha, true, x, incX, A, ldA, 1);
    }
}

void
ULMBLAS(cher_)(enum CBLAS_UPLO    upLo,
               ULMBLAS_INDEX_TYPE n,
               float              alpha,
               const float        *x_,
               ULMBLAS_INDEX_TYPE incX,
               float              *A_,
               ULMBLAS_INDEX_TYPE ldA)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *A = reinterpret_cast<fcomplex *>(A_);
//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (upLo==CblasLower) {
        ulmBLAS::helr<std::ptrdiff_t>(n, alpha, true, x, incX, A, 1, ldA);
    } else {
        ulmBLAS::helr<std::ptrdiff_t>(n, alpha, false, x, incX, A, ldA, 1);
    }
}

void
ULMBLAS(zher)(enum CBLAS_UPLO    upLo,
              ULMBLAS_INDEX_TYPE n,
              double             alpha,
              const double       *x_,
              ULMBLAS_INDEX_TYPE incX,
              double             *A_,
              ULMBLAS_INDEX_TYPE ldA)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *A = reinterpret_cast<dcomplex *>(A_);
//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (upLo==CblasLower) {
        ulmBLAS::helr<std::ptrdiff_t>(n, alpha, false, x, incX, A, 1, ldA);
    } else {
        ulmBLAS::helr<std::ptrdiff_t>(n, alpha, true, x, incX, A, ldA, 1);
    }
}

void
ULMBLAS(zher_)(enum CBLAS_UPLO    upLo,
               ULMBLAS_INDEX_TYPE n,
               double             alpha,
               const double       *x_,
               ULMBLAS_INDEX_TYPE incX,
               double             *A_,
               ULMBLAS_INDEX_TYPE ldA)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *A = reinterpret_cast<dcomplex *>(A_);
//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (upLo==CblasLower) {
        ulmBLAS::helr<std::ptrdiff_t>(n, alpha, true, x, incX, A, 1, ldA);
    } else {
        ulmBLAS::helr<std::ptrdiff_t>(n, alpha, false, x, incX, A, ldA, 1);
    }
}

void
CBLAS(cher)(enum CBLAS_ORDER   order,
            enum CBLAS_UPLO    upLo,
            ULMBLAS_INDEX_TYPE n,
            float              alpha,
            const void         *x_v,
            ULMBLAS_INDEX_TYPE incX,
            void               *A_v,
            ULMBLAS_INDEX_TYPE ldA)
{
    const float *x = static_cast<const float *>(x_v);
    float *A = static_cast<float *>(A_v);

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (upLo!=CblasUpper && upLo!=CblasLower) {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (incX==0) {
        info = 6;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 8;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_cher", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(cher)(upLo, n, alpha, x, incX, A, ldA);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(cher_)(upLo, n, alpha, x, incX, A, ldA);
    }

}

void
CBLAS(zher)(enum CBLAS_ORDER   order,
            enum CBLAS_UPLO    upLo,
            ULMBLAS_INDEX_TYPE n,
            double             alpha,
            const void         *x_v,
            ULMBLAS_INDEX_TYPE incX,
            void               *A_v,
            ULMBLAS_INDEX_TYPE ldA)
{
    const double *x = static_cast<const double *>(x_v);
    double *A = static_cast<double *>(A_v);

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (upLo!=CblasUpper && upLo!=CblasLower) {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (incX==0) {
        info = 6;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 8;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_zher", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(zher)(upLo, n, alpha, x, incX, A, ldA);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(zher_)(upLo, n, alpha, x, incX, A, ldA);
    }

}

} // extern "C"
