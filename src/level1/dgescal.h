// Compute X *= alpha

#ifndef LEVEL1_DGESCAL
#define LEVEL1_DGESCAL 1

#include <ulmblas.h>

void
ULMBLAS(dgescal)(int     m,
                 int     n,
                 double  alpha,
                 double  *X,
                 int     incRowX,
                 int     incColX);

void
dgescal(int     m,
        int     n,
        double  alpha,
        double  *X,
        int     incRowX,
        int     incColX);

#endif // LEVEL1_DGESCAL
