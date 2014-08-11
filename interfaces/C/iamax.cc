#include <interfaces/C/config.h>
#include <src/level1/iamax.h>
#include <src/level1/iamax.tcc>

extern "C" {

int
ULMBLAS(idamax)(const int       n,
                const double    *x,
                const int       incX)
{
    return ulmBLAS::iamax(n, x, incX);
}

} // extern "C"
