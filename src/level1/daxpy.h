#ifndef ULMBLAS_SRC_LEVEL1_DAXPY_H
#define ULMBLAS_SRC_LEVEL1_DAXPY_H 1

void
daxpy(const int     n,
      const double  alpha,
      const double  *x,
      const int     incX,
      double        *y,
      int           incY);

#endif // ULMBLAS_SRC_LEVEL1_DAXPY_H
