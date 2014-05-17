#include <ulmblas.h>
#include <math.h>

static double
sign(const double x, const double y)
{
    if (y>=0.0) {
        return fabs(x);
    } else {
        return -fabs(x);
    }
}

void
ULMBLAS(drotg)(double *a,
               double *b,
               double *c,
               double *s)
{
    double r, roe = *b, scale, z;

    if (fabs(*a)>fabs(*b)) {
        roe = *a;
    }
    scale = fabs(*a) + fabs(*b);
    if (scale==0.0) {
        *c = 1.0;
        *s = 0.0;
        r = 0.0;
        z = 0.0;
    } else {
        r = scale*sqrt(pow(*a/scale, 2.0) + pow(*b/scale, 2.0));
        r = sign(1.0, roe)*r;
        *c = *a / r;
        *s = *b / r;
        z = 1.0;
        if (fabs(*a)>fabs(*b)) {
            z = *s;
        }
        if (fabs(*b)>=fabs(*a) && *c!=0.0) {
            z = 1.0/(*c);
        }
    }
    *a = r;
    *b = z;
}

void
F77BLAS(drotg)(double *a,
               double *b,
               double *c,
               double *s)
{
    ULMBLAS(drotg)(a, b, c, s);
}
