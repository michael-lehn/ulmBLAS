#include <ulmblas.h>
#include <stdio.h>

#define M 4
#define K 4
#define N 4

double A[M*K];
double B[K*N];
double C[M*N];

void
initMatrix(int m, int n, double *X, int ldX, int counter)
{
    int i, j;

    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
            X[i+j*ldX] = counter++;
        }
    }
}

void
printMatrix(int m, int n, const double *X, int ldX)
{
    int i, j;

    for (i=0; i<m; ++i) {
        for (j=0; j<n; ++j) {
            printf("  %4.0lf", X[i+j*ldX]);
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
    initMatrix(M, K, A, M, 1);
    initMatrix(K, N, B, K, M*K+1);
    initMatrix(M, N, C, M, M*K+K*N+1);

    printf("A = \n");
    printMatrix(M, K, A, M);
    printf("B = \n");
    printMatrix(K, N, B, K);

    ULMBLAS(dgemm)(NoTrans, NoTrans,
                   M, N, K,
                   1.0,
                   A, M,
                   B, K,
                   0.0,
                   C, M);


    printf("C = \n");
    printMatrix(M, N, C, M);

    return 0;
}
