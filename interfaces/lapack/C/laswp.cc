#include LAPACK_HEADER
#include <ulmblas/lapack/laswp.h>

extern "C" {

void
ULMLAPACK(dlaswp)(ULMBLAS_INDEX_TYPE n,
                  double             *A,
                  ULMBLAS_INDEX_TYPE ldA,
                  ULMBLAS_INDEX_TYPE k1,
                  ULMBLAS_INDEX_TYPE k2,
                  ULMBLAS_INDEX_TYPE *piv,
                  ULMBLAS_INDEX_TYPE incPiv)
{
    ulmBLAS::laswp<std::ptrdiff_t>(n, A, 1, ldA, k1, k2, piv, incPiv);
}
} // extern "C"

