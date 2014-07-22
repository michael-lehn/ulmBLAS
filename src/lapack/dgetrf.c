#include <stdio.h>
#include <ulmblas.h>

//
//  Declarations for needed BLAS Level 1 functions
//
int
ULMBLAS(idamax)(const int       n,
                const double    *x,
                const int       incX);

void
ULMBLAS(dswap)(const int n,
               double    *x,
               const int incX,
               double    *y,
               const int incY);

//
//  Unblocked implementation of the LU factorization
//
int
dgetrf_unblk(int     m,
             int     n,
             double  *A,
             int     incRowA,
             int     incColA,
             int     *piv)
{
    int i, j, jp, info;

    info = 0;

    if (m==0 || n==0) {
        return info;
    }

    for (j=0; j<m && j<n; ++j) {
        jp = j + ULMBLAS(idamax)(m-j, &A[j*incRowA+j*incColA], incRowA);
        piv[j] = jp+1;
        if (A[jp*incRowA+j*incColA]!=0.0) {
            if (jp!=j) {
                ULMBLAS(dswap)(n,
                               &A[j*incRowA], incColA,
                               &A[jp*incRowA], incColA);
            }
        } else {
            if (info==0) {
                info = j+1;
            }
        }

    }

    for (i=0; i<m; ++i) {
        piv[i]=i+1;
    }
    return 0;
}

int
ULMBLAS(dgetrf)(enum Order  order,
                int         m,
                int         n,
                double      *A,
                int         ldA,
                int         *piv)
{
    if (order==ColMajor) {
        return dgetrf_unblk(m, n, A, 1, ldA, piv);
    } else {
        return dgetrf_unblk(m, n, A, ldA, 1, piv);
    }

    return 0;
}
