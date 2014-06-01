#include <stdio.h>

#define M   14
#define K   15

#define MC  8
#define KC  12

#define MR  4

double  A[M*K];
double _A[MC*KC];

void
initMatrix(int m, int n, double *X, int ldX, int counter)
{
    int i, j;

    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
            X[i+j*ldX] = counter;
            ++counter;
        }
    }
}

void
printMatrix(int m, int n, double *X, int ldX)
{
    int i, j;

    for (i=0; i<m; ++i) {
        for (j=0; j<n; ++j) {
            printf("%4.0lf ", X[i+j*ldX]);
        }
        printf("\n");
    }
    printf("\n");
}

void
pack_MRxk(int n, double  *A, int incRowA, int incColA, double *buffer)
{
    int i, j;

    for (j=0; j<n; ++j) {
        for (i=0; i<MR; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += MR;
        A      += incColA;
    }
}

int
main()
{
    initMatrix(M, K, A, M, 1);

    printf("A = \n");
    printMatrix(M, K, A, M);

    pack_MRxk(KC, A, 1, M, _A);

    printf("Buffer: _A = \n");
    printMatrix(MC, KC, _A, MC);

    pack_MRxk(KC, A+MR, 1, M, _A+MR*KC);

    printf("Buffer: _A = \n");
    printMatrix(MC, KC, _A, MC);
    return 0;
}
