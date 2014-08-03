#ifndef ULMBLAS_SRC_LAPACK_DLASWP_H
#define ULMBLAS_SRC_LAPACK_DLASWP_H 1

void
dlaswp(int n,
       double *A, int incRowA, int incColA,
       int k1, int k2,
       int *piv, int incPiv);

void
ULMBLAS(dlaswp)(int n,
                double *A, int ldA,
                int k1, int k2,
                int *piv, int incPiv);

#endif // ULMBLAS_SRC_LAPACK_DLASWP_H
