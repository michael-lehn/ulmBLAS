void
drot_(const int      *_n,
      double         *x,
      const int      *_incX,
      double         *y,
      const int      *_incY,
      const double   *_c,
      const double   *_s)
{
//
//  Dereference scalar parameters
//
    int n    = *_n;
    int incX = *_incX;
    int incY = *_incY;
    double c = *_c;
    double s = *_s;

//
//  Local scalars
//
    int    i;
    double tmp;

//
//  Quick return if possible
//
    if (n==0) {
        return;
    }
    if (incX==1 && incY==1) {
//
//      Code for both increments equal to 1
//
        for (i=0; i<n; ++i) {
            tmp  = c*x[i] + s*y[i];
            y[i] = c*y[i] - s*x[i];
            x[i] = tmp;
        }
    } else {
//
//      Code for unequal increments or equal increments not equal to 1
//
        if (incX<0) {
            x -= incX*(n-1);
        }
        if (incY<0) {
            y -= incY*(n-1);
        }
        for (i=0; i<n; ++i, x+=incX, y+=incY) {
            tmp  = c*(*x) + s*(*y);
            (*y) = c*(*y) - s*(*x);
            (*x) = tmp;
        }
    }
}
