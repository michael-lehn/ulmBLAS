#include <ulmblas.h>
#include <stdio.h>

#define M 4
#define N 5

int    piv[M];
double A[M*N];

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
            printf("  %6.3lf", X[i+j*ldX]);
        }
        printf("\n");
    }
    printf("\n");
}

void
printIntMatrix(int m, int n, const int *X, int ldX)
{
    int i, j;

    for (i=0; i<m; ++i) {
        for (j=0; j<n; ++j) {
            printf("  %4d", X[i+j*ldX]);
        }
        printf("\n");
    }
    printf("\n");
}

// as we know what we are doing: quick and dirty declaration  ;-)
int
ULMBLAS(dgetrf)();

int
main()
{
    int info;

    initMatrix(M, N, A, M, 1);

    printf("A = \n");
    printMatrix(M, N, A, M);

    info = ULMBLAS(dgetrf)(M, N,
                           A, M,
                           piv);

    printf("A = \n");
    printMatrix(M, N, A, M);

    printf("piv = \n");
    printIntMatrix(1, M, piv, 1);

    printf("info = %d\n", info);

    return 0;
}
