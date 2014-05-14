#include <ulmblas.h>
#include <math.h>

int
ULMBLAS(idamax)(const int       n,
                const double    *x,
                const int       incX)
{
//
//  Local scalars
//
    int    i, iamax = 0;
    double amax;

//
//  Quick return if possible
//
    if (n<1 || incX<=0) {
        return 0;
    }
    if (n==1) {
        return 1;
    }
    if (incX==1) {
//
//      Code for increment equal to 1
//
        amax = fabs(x[0]);
        for (i=1; i<n; ++i) {
            if (fabs(x[i])>amax) {
                iamax = i;
                amax  = fabs(x[i]);
            }
        }
    } else {
//
//      Code for increment not equal to 1
//
        iamax = 0;
        amax  = fabs(x[0]);
        for (i=1, x+=incX; i<n; ++i, x+=incX) {
            if (fabs(*x)>amax) {
                iamax = i;
                amax  = fabs(*x);
            }
        }
    }
    return iamax+1;
}

int
F77BLAS(idamax)(const int       *_n,
                const double    *x,
                const int       *_incX)
{
//
//  Dereference scalar parameters
//
    int n    = *_n;
    int incX = *_incX;

    return ULMBLAS(idamax)(n, x, incX);
}
