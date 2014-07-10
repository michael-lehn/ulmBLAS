// Compute Y += alpha*X

#ifndef LEVEL1_DGEAXPY
#define LEVEL1_DGEAXPY 1

#include <ulmblas.h>

void
ULMBLAS(dgeaxpy)(int     m,
                 int     n,
                 double  alpha,
                 double  *X,
                 int     incRowX,
                 int     incColX,
                 double  *Y,
                 int     incRowY,
                 int     incColY);

#endif // LEVEL1_DGEAXPY
