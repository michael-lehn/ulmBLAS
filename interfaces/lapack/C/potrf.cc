#include <complex>
#include <cstddef>
#include LAPACK_HEADER
#include <ulmblas/lapack/potrf.h>

//
//  Cholesky factorization (ATLAS' clapack_?potrf) of the matrix as stored
//  (column or row major): upLo CblasLower: A = L L^H, CblasUpper: A = U^H U
//  (L^T, U^T for real types).  Only that triangle is referenced.  Returns
//  LAPACK's INFO: 0, or i if the leading minor of order i is not positive
//  definite.  Invalid arguments: -i for argument i.
//

namespace {

template <typename T>
ULMBLAS_INDEX_TYPE
potrf(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n, T *A,
      ULMBLAS_INDEX_TYPE ldA)
{
    if (order != CblasColMajor && order != CblasRowMajor) {
	return -1;
    }
    if (upLo != CblasLower && upLo != CblasUpper) {
	return -2;
    }
    if (n < 0) {
	return -3;
    }
    if (ldA < (n > 1 ? n : 1)) {
	return -5;
    }
    const std::ptrdiff_t incRow = (order == CblasColMajor) ? 1 : ldA;
    const std::ptrdiff_t incCol = (order == CblasColMajor) ? ldA : 1;
    return ulmBLAS::potrf<std::ptrdiff_t>(upLo == CblasLower, n, A, incRow,
                                          incCol);
}

} // namespace

extern "C" {

ULMBLAS_INDEX_TYPE
ULMLAPACK(spotrf)(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
                  ULMBLAS_INDEX_TYPE n, float *A, ULMBLAS_INDEX_TYPE ldA)
{
    return potrf(order, upLo, n, A, ldA);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(dpotrf)(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
                  ULMBLAS_INDEX_TYPE n, double *A, ULMBLAS_INDEX_TYPE ldA)
{
    return potrf(order, upLo, n, A, ldA);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(cpotrf)(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
                  ULMBLAS_INDEX_TYPE n, void *A, ULMBLAS_INDEX_TYPE ldA)
{
    return potrf(order, upLo, n, static_cast<std::complex<float> *>(A), ldA);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(zpotrf)(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
                  ULMBLAS_INDEX_TYPE n, void *A, ULMBLAS_INDEX_TYPE ldA)
{
    return potrf(order, upLo, n, static_cast<std::complex<double> *>(A), ldA);
}

} // extern "C"
