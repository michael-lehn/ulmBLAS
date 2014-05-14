double
ddot_(const int     *_n,
      const double  *x,
      const int     *_incX,
      const double  *y,
      const int     *_incY)
{
    int n    = *_n;
    int incX = *_incX;
    int incY = *_incY;

    int    i;
    double result = 0.0;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    for (i=0; i<n; ++i, x+=incX, y+=incY) {
        result += (*x) * (*y);
    }

    return result;
}
