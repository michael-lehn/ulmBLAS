#include <math.h>

double
dnrm2_(const int     *_n,
       const double  *x,
       const int     *_incX)
{
//
//  Dereference scalar parameters
//
    int n    = *_n;
    int incX = *_incX;

//
//  Local scalars
//
    int    i;
    double result, scale, ssq, absX;

    if (n<1 || incX<1) {
        result = 0.0;
    } else if (n==1) {
        result = fabs(x[0]);
    } else {
        scale = 0.0;
        ssq   = 1.0;
        for (i=0; i<1+(n-1)*incX; i+=incX) {
            if (x[i]!=0.0) {
                absX = fabs(x[i]);
                if (scale<absX) {
                    ssq = 1.0 + ssq*pow(scale/absX, 2.0);
                    scale = absX;
                } else {
                    ssq += pow(absX/scale, 2.0);
                }
            }
        }
        result = scale*sqrt(ssq);
    }

    return result;
}
