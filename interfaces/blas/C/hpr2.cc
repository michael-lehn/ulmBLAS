#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(chpr2)(enum CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n, const float *alpha_,
               const float *x_, ULMBLAS_INDEX_TYPE incX, const float *y_,
               ULMBLAS_INDEX_TYPE incY, float *A_)
{
    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);
    fcomplex *A = reinterpret_cast<fcomplex *>(A_);

    if (incX < 0) {
	x -= incX * (n - 1);
    }
    if (incY < 0) {
	y -= incY * (n - 1);
    }
    //
    //  Start the operations.
    //
    if (upLo == CblasLower) {
	ulmBLAS::hplr2<std::ptrdiff_t>(n, alpha, x, incX, y, incY, A);
    } else {
	ulmBLAS::hpur2<std::ptrdiff_t>(n, alpha, x, incX, y, incY, A);
    }
}

void
ULMBLAS(chpr2_)(enum CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n, const float *alpha_,
                const float *x_, ULMBLAS_INDEX_TYPE incX, const float *y_,
                ULMBLAS_INDEX_TYPE incY, float *A_)
{
    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);
    fcomplex *A = reinterpret_cast<fcomplex *>(A_);

    if (incX < 0) {
	x -= incX * (n - 1);
    }
    if (incY < 0) {
	y -= incY * (n - 1);
    }
    //
    //  Start the operations.
    //
    if (upLo == CblasLower) {
	ulmBLAS::hplr2<std::ptrdiff_t>(n, true, alpha, x, incX, y, incY, A);
    } else {
	ulmBLAS::hpur2<std::ptrdiff_t>(n, true, alpha, x, incX, y, incY, A);
    }
}

void
ULMBLAS(zhpr2)(enum CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n, const double *alpha_,
               const double *x_, ULMBLAS_INDEX_TYPE incX, const double *y_,
               ULMBLAS_INDEX_TYPE incY, double *A_)
{
    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);
    dcomplex *A = reinterpret_cast<dcomplex *>(A_);

    if (incX < 0) {
	x -= incX * (n - 1);
    }
    if (incY < 0) {
	y -= incY * (n - 1);
    }
    //
    //  Start the operations.
    //
    if (upLo == CblasLower) {
	ulmBLAS::hplr2<std::ptrdiff_t>(n, alpha, x, incX, y, incY, A);
    } else {
	ulmBLAS::hpur2<std::ptrdiff_t>(n, alpha, x, incX, y, incY, A);
    }
}

void
ULMBLAS(zhpr2_)(enum CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                const double *alpha_, const double *x_, ULMBLAS_INDEX_TYPE incX,
                const double *y_, ULMBLAS_INDEX_TYPE incY, double *A_)
{
    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);
    dcomplex *A = reinterpret_cast<dcomplex *>(A_);

    if (incX < 0) {
	x -= incX * (n - 1);
    }
    if (incY < 0) {
	y -= incY * (n - 1);
    }
    //
    //  Start the operations.
    //
    if (upLo == CblasLower) {
	ulmBLAS::hplr2<std::ptrdiff_t>(n, true, alpha, x, incX, y, incY, A);
    } else {
	ulmBLAS::hpur2<std::ptrdiff_t>(n, true, alpha, x, incX, y, incY, A);
    }
}

void
CBLAS(chpr2)(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
             const void *alpha_v, const void *x_v, ULMBLAS_INDEX_TYPE incX,
             const void *y_v, ULMBLAS_INDEX_TYPE incY, void *A_v)
{
    const float *alpha = static_cast<const float *>(alpha_v);
    const float *x = static_cast<const float *>(x_v);
    const float *y = static_cast<const float *>(y_v);
    float *A = static_cast<float *>(A_v);

    //
    //  Test the input parameters
    //
    int info = 0;

    if (order != CblasColMajor && order != CblasRowMajor) {
	info = 1;
    } else if (upLo != CblasUpper && upLo != CblasLower) {
	info = 2;
    } else if (n < 0) {
	info = 3;
    } else if (incX == 0) {
	info = (order == CblasColMajor) ? 6 : 8;
    } else if (incY == 0) {
	info = (order == CblasColMajor) ? 8 : 6;
    }

    if (info != 0) {
	extern int RowMajorStrg;

	RowMajorStrg = (order == CblasRowMajor) ? 1 : 0;
	CBLAS(xerbla)(info, "cblas_chpr2", "... bla bla ...");
	return;
    }

    if (order == CblasColMajor) {
	ULMBLAS(chpr2)(upLo, n, alpha, x, incX, y, incY, A);
    } else {
	upLo = (upLo == CblasUpper) ? CblasLower : CblasUpper;
	ULMBLAS(chpr2_)(upLo, n, alpha, x, incX, y, incY, A);
    }
}

void
CBLAS(zhpr2)(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
             const void *alpha_v, const void *x_v, ULMBLAS_INDEX_TYPE incX,
             const void *y_v, ULMBLAS_INDEX_TYPE incY, void *A_v)
{
    const double *alpha = static_cast<const double *>(alpha_v);
    const double *x = static_cast<const double *>(x_v);
    const double *y = static_cast<const double *>(y_v);
    double *A = static_cast<double *>(A_v);

    //
    //  Test the input parameters
    //
    int info = 0;

    if (order != CblasColMajor && order != CblasRowMajor) {
	info = 1;
    } else if (upLo != CblasUpper && upLo != CblasLower) {
	info = 2;
    } else if (n < 0) {
	info = 3;
    } else if (incX == 0) {
	info = (order == CblasColMajor) ? 6 : 8;
    } else if (incY == 0) {
	info = (order == CblasColMajor) ? 8 : 6;
    }

    if (info != 0) {
	extern int RowMajorStrg;

	RowMajorStrg = (order == CblasRowMajor) ? 1 : 0;
	CBLAS(xerbla)(info, "cblas_zhpr2", "... bla bla ...");
	return;
    }

    if (order == CblasColMajor) {
	ULMBLAS(zhpr2)(upLo, n, alpha, x, incX, y, incY, A);
    } else {
	upLo = (upLo == CblasUpper) ? CblasLower : CblasUpper;
	ULMBLAS(zhpr2_)(upLo, n, alpha, x, incX, y, incY, A);
    }
}

} // extern "C"
