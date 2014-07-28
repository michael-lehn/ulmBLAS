#ifndef ULMBLAS_SRC_LAPACK_DGETRF_H
#define ULMBLAS_SRC_LAPACK_DGETRF_H 1

int
dgetf2(int     m,
       int     n,
       double  *A,
       int     incRowA,
       int     incColA,
       int     *piv);

int
dgetrf(int     m,
       int     n,
       double  *A,
       int     incRowA,
       int     incColA,
       int     *piv);

int
ULMBLAS(dgetrf)(enum Order  order,
                int         m,
                int         n,
                double      *A,
                int         ldA,
                int         *piv);

#endif // ULMBLAS_SRC_LAPACK_DGETRF_H
