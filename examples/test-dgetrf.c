#include <stdio.h>
#include <ulmblas.h>

static void
printMatrix(int m, int n, const double *X, int incRowX, int incColX)
{
    int i, j;

    for (i=0; i<m; ++i) {
        for (j=0; j<n; ++j) {
            printf("  %9.4lf", X[i*incRowX+j*incColX]);
        }
        printf("\n");
    }
    printf("\n");
}

static void
initMatrix(int m, int n, double *X, int incRowX, int incColX, int count)
{
    int i, j;

    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
            X[i*incRowX+j*incColX] = count++;
        }
        count *= (j+1);
    }
}

int
ULMBLAS(dgetrf)();

#define M 6
#define N 6

double A[M*N];
int    piv[min(M,N)];


int
main()
{
    int i, info;

    initMatrix(M, N, A, 1, M, 1);

    printf("A = \n");
    printMatrix(M, N, A, 1, M);

    info = ULMBLAS(dgetrf)(ColMajor, M, N, A, M,piv);

    printf("info = %d\n", info);

    printf("LU = \n");
    printMatrix(M, N, A, 1, M);

    for (i=0; i<min(M,N); ++i) {
        printf("piv(%d) = %d\n", i, piv[i]);
    }

    return 0;
}
