




#ifndef CLAPACK
#define CLAPACK 1

#include <atlblas.h>

#ifdef __cplusplus
extern "C" {
#endif

int
ATL_dgetrf(enum Order order,
                int m,
                int n,
                double *A,
                int ldA,
                int *piv);

void
ATL_dlaswp(int n,
                  double *A,
                  int ldA,
                  int k1,
                  int k2,
                  int *piv,
                  int incPiv);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CBLAS
