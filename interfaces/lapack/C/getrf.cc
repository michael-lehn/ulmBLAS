#include <complex>
#include <cstddef>
#include LAPACK_HEADER
#include <ulmblas/lapack/getrf.h>

//
//  LU factorization with partial pivoting (ATLAS' clapack_?getrf): pivot
//  indices are 0-based, the return value is LAPACK's 1-based INFO.  For row
//  major storage the matrix is factored as its transpose in column major
//  storage (A P = L U with column interchanges, L lower and U unit upper
//  triangular), as ATLAS does.  Invalid arguments: -i for argument i.
//

namespace {

template <typename T>
ULMBLAS_INDEX_TYPE
getrf(enum CBLAS_ORDER order, ULMBLAS_INDEX_TYPE m, ULMBLAS_INDEX_TYPE n, T *A,
      ULMBLAS_INDEX_TYPE ldA, ULMBLAS_INDEX_TYPE *piv)
{
    if (order != CblasColMajor && order != CblasRowMajor) {
	return -1;
    }
    if (m < 0) {
	return -2;
    }
    if (n < 0) {
	return -3;
    }
    const ULMBLAS_INDEX_TYPE rows = (order == CblasColMajor) ? m : n;
    if (ldA < (rows > 1 ? rows : 1)) {
	return -5;
    }
    if (order == CblasColMajor) {
	return ulmBLAS::getrf<std::ptrdiff_t>(m, n, A, 1, ldA, piv, 1);
    } else {
	return ulmBLAS::getrf<std::ptrdiff_t>(n, m, A, 1, ldA, piv, 1);
    }
}

} // namespace

extern "C" {

ULMBLAS_INDEX_TYPE
ULMLAPACK(sgetrf)(enum CBLAS_ORDER order, ULMBLAS_INDEX_TYPE m,
                  ULMBLAS_INDEX_TYPE n, float *A, ULMBLAS_INDEX_TYPE ldA,
                  ULMBLAS_INDEX_TYPE *piv)
{
    return getrf(order, m, n, A, ldA, piv);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(dgetrf)(enum CBLAS_ORDER order, ULMBLAS_INDEX_TYPE m,
                  ULMBLAS_INDEX_TYPE n, double *A, ULMBLAS_INDEX_TYPE ldA,
                  ULMBLAS_INDEX_TYPE *piv)
{
    return getrf(order, m, n, A, ldA, piv);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(cgetrf)(enum CBLAS_ORDER order, ULMBLAS_INDEX_TYPE m,
                  ULMBLAS_INDEX_TYPE n, void *A, ULMBLAS_INDEX_TYPE ldA,
                  ULMBLAS_INDEX_TYPE *piv)
{
    return getrf(order, m, n, static_cast<std::complex<float> *>(A), ldA, piv);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(zgetrf)(enum CBLAS_ORDER order, ULMBLAS_INDEX_TYPE m,
                  ULMBLAS_INDEX_TYPE n, void *A, ULMBLAS_INDEX_TYPE ldA,
                  ULMBLAS_INDEX_TYPE *piv)
{
    return getrf(order, m, n, static_cast<std::complex<double> *>(A), ldA, piv);
}

} // extern "C"
