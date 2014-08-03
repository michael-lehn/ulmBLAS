#ifndef ULMBLAS_SRC_LEVEL1_DGER_H
#define ULMBLAS_SRC_LEVEL1_DGER_H 1

void
dger(const int     m,
     const int     n,
     const double  alpha,
     const double  *x,
     const int     incX,
     const double  *y,
     const int     incY,
     double        *A,
     const int     incRowA,
     const int     incColA);

#endif // ULMBLAS_SRC_LEVEL1_DGER_H
