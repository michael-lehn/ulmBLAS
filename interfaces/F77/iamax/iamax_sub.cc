#include <interfaces/F77/config.h>
#include <src/level1/iamax.h>

extern "C" {

void
F77BLAS(idamax_sub)(const int       *_n,
                    const double    *x,
                    const int       *_incX,
                    int             *_result)
{
//
//  Dereference scalar parameters
//
    int n    = *_n;
    int incX = *_incX;

    if (incX<0) {
        x -= incX*(n-1);
    }
    *_result = ulmBLAS::iamax(n, x, incX)+1;
}

} // extern "C"
