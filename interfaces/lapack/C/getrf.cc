#include LAPACK_HEADER
#include <ulmblas/lapack/getrf.h>

extern "C" {

ULMBLAS_INDEX_TYPE
ULMLAPACK(dgetrf)(enum CBLAS_ORDER   order,
                  ULMBLAS_INDEX_TYPE m,
                  ULMBLAS_INDEX_TYPE n,
                  double             *A,
                  ULMBLAS_INDEX_TYPE ldA,
                  ULMBLAS_INDEX_TYPE *piv)
{
    if (order==CblasColMajor) {
        return ulmBLAS::getrf<std::ptrdiff_t>(m, n, A, 1, ldA, piv, 1);
    } else {
        return ulmBLAS::getrf<std::ptrdiff_t>(n, m, A, 1, ldA, piv, 1);
    }
}

} // extern "C"
