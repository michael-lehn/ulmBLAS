#include BLAS_HEADER
#include <ulmblas/level1/iamax.h>

extern "C" {

int
ULMBLAS(idamax)(const int       n,
                const double    *x,
                const int       incX)
{
    return ulmBLAS::iamax(n, x, incX);
}

int
CBLAS(idamax)(const int       n,
              const double    *x,
              const int       incX)
{
    return ulmBLAS::iamax(n, x, incX);
}

} // extern "C"
