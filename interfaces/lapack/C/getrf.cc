#include LAPACK_HEADER
#include <ulmblas/lapack/getrf.h>

extern "C" {

int
ULMLAPACK(dgetrf)(enum CBLAS_ORDER  order,
                  int               m,
                  int               n,
                  double            *A,
                  int               ldA,
                  int               *piv)
{
    if (order==CblasColMajor) {
        return ulmBLAS::getrf(m, n, A, 1, ldA, piv, 1);
    } else {
        return ulmBLAS::getrf(n, m, A, ldA, 1, piv, 1);
    }
}

} // extern "C"
