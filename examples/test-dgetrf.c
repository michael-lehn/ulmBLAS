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

#define M 3
#define N 3

double A[M*N];
int    piv[min(M,N)];


int
main()
{
    int i, info;

    //initMatrix(M, N, A, 1, M, 1);
    A[0] = 1; A[3] = 2; A[6] = 3;
    A[1] = 4; A[4] = 5; A[7] = 6;
    A[2] = 7; A[5] = 8; A[8] = 1;

    printf("A = \n");
    printMatrix(M, N, A, 1, M);

    info = ULMBLAS(dgetrf)(ColMajor, M, N, A, M,piv);

    printf("info = %d\n", info);

    printf("LU = \n");
    printMatrix(M, N, A, 1, M);

    for (i=0; i<min(M,N); ++i) {
        printf("piv(%d) = %d\n", i, piv[i]);
    }

    A[0] = 1; A[1] = 2; A[2] = 3;
    A[3] = 4; A[4] = 5; A[5] = 6;
    A[6] = 7; A[7] = 8; A[8] = 1;

    printf("A = \n");
    printMatrix(M, N, A, M, 1);

    info = ULMBLAS(dgetrf)(RowMajor, M, N, A, M,piv);

    printf("info = %d\n", info);

    printf("LU = \n");
    printMatrix(M, N, A, M, 1);

    for (i=0; i<min(M,N); ++i) {
        printf("piv(%d) = %d\n", i, piv[i]);
    }


    A[0]=  0.333622;  A[3]= -0.489840;  A[6]=  0.459556;
    A[1]= -0.402201;  A[4]=  0.302918;  A[7]=  0.410026;
    A[2]=  0.149139;  A[5]=  0.121483;  A[8]= -0.367779;

    printf("A = \n");
    printMatrix(M, N, A, 1, M);

    info = ULMBLAS(dgetrf)(ColMajor, M, N, A, M,piv);

    printf("info = %d\n", info);

    printf("LU = \n");
    printMatrix(M, N, A, 1, M);

    for (i=0; i<min(M,N); ++i) {
        printf("piv(%d) = %d\n", i, piv[i]);
    }


    A[0]=  0.333622;  A[1]= -0.489840;  A[2]=  0.459556;
    A[3]= -0.402201;  A[4]=  0.302918;  A[5]=  0.410026;
    A[6]=  0.149139;  A[7]=  0.121483;  A[8]= -0.367779;

    printf("A = \n");
    printMatrix(M, N, A, M, 1);

    info = ULMBLAS(dgetrf)(ColMajor, M, N, A, M, piv);

    printf("info = %d\n", info);

    printf("LU = \n");
    printMatrix(M, N, A, M, 1);

    for (i=0; i<min(M,N); ++i) {
        printf("piv(%d) = %d\n", i, piv[i]);
    }


    return 0;
}
