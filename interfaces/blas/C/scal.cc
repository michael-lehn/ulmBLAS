#include BLAS_HEADER
#include <ulmblas/level1/scal.h>

extern "C" {

void
ULMBLAS(dscal)(const int    n,
               const double alpha,
               double       *x,
               const int    incX)
{
    return ulmBLAS::scal(n, alpha, x, incX);
}

} // extern "C"
