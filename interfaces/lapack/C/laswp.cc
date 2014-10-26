#include <interfaces/blas/C/config.h>
#include <ulmblas/lapack/laswp.h>

extern "C" {

void
ULMBLAS(dlaswp)(int     n,
                double  *A,
                int     ldA,
                int     k1,
                int     k2,
                int     *piv,
                int     incPiv)
{
    ulmBLAS::laswp(n, A, 1, ldA, k1, k2, piv, incPiv);
}
} // extern "C"

