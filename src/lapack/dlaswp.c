#include <ulmblas.h>

void
dlaswp(int n,
       double *A, int incRowA, int incColA,
       int k1, int k2,
       int *piv, int incPiv)
{
    int    i, i1, i2, inc, ip, j, k, n32;
    double tmp;

    if (incPiv>0) {
        i1  = k1;
        i2  = k2-1;
        inc = 1;
    } else if (incPiv<0) {
        i1  = k2-1;
        i2  = k1;
        inc = -1;
    } else {
        return;
    }

    n32 = (n/32)*32;
    if (n32!=0) {
        for (j=0; j<n32; j+=32) {
            for (i=i1; i<=i2; i+=inc) {
                ip = piv[i*incPiv];
                if (ip!=i) {
                    for (k=j; k<j+32; ++k) {
                        tmp                     = A[i *incRowA+k*incColA];
                        A[i *incRowA+k*incColA] = A[ip*incRowA+k*incColA];
                        A[ip*incRowA+k*incColA] = tmp;
                    }
                }
            }
        }
    }
    if (n32!=n) {
        for (i=i1; i<=i2; i+=inc) {
            ip = piv[i*incPiv];
            if (ip!=i) {
                for (k=n32; k<n; ++k) {
                    tmp                     = A[i *incRowA+k*incColA];
                    A[i *incRowA+k*incColA] = A[ip*incRowA+k*incColA];
                    A[ip*incRowA+k*incColA] = tmp;
                }
            }
        }
    }
}

void
ULMBLAS(dlaswp)(int n,
                double *A, int ldA,
                int k1, int k2,
                int *piv, int incPiv)
{
    dlaswp(n, A, 1, ldA, k1, k2, piv, incPiv);
}
