#include <ulmblas.h>
#include <stdio.h>

#define M 4
#define K 1
#define N 4

double A[M*K];
double B[K*N];
double C[M*N];

void
printMatrix(int m, int n, const double *X, int ldX)
{
    int i, j;

    for (i=0; i<m; ++i) {
        for (j=0; j<n; ++j) {
            printf("  %+8.6lf", X[i+j*ldX]);
        }
        printf("\n");
    }
    printf("\n");
}

// as we know what we are doing: quick and dirty declaration  ;-)
void
ULMBLAS(dgemm)();

int
main()
{
    A[0+0*M] =  0.386613;
    A[1+0*M] = -0.182817;
    A[2+0*M] = -0.052947;
    A[3+0*M] =  0.000000;

    B[0+0*K] =  0.306693;
    B[0+1*K] = -0.462537;
    B[0+2*K] =  0.466533;
    B[0+3*K] =  0.000000;

    printf("-> A = \n");
    printMatrix(M, K, A, M);
    printf("-> B = \n");
    printMatrix(K, N, B, K);

    ULMBLAS(dgemm)(NoTrans, NoTrans,
                   M, N, K,
                   1.0,
                   A, M,
                   B, K,
                   0.0,
                   C, M);


    printf("-> C = \n");
    printMatrix(M, N, C, M);

    return 0;
}
