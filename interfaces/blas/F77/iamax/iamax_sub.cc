#include <interfaces/blas/F77/config.h>
#include <ulmblas/level1/iamax.h>

extern "C" {

void
F77BLAS(idamax_sub)(const int       *n_,
                    const double    *x,
                    const int       *incX_,
                    int             *result_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    *result_ = ulmBLAS::iamax(n, x, incX)+1;
}

} // extern "C"
