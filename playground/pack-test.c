#include <assert.h>
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
pack_MRxk(int k, double  *A, int incRowA, int incColA, double *buffer)
{
    int i, j;

    for (j=0; j<k; ++j) {
        for (i=0; i<MR; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += MR;
        A      += incColA;
    }
}

void
pack_A(int mc, int kc, double *A, int incRowA, int incColA, double *buffer)
{
    const int MP  = mc / MR;
    const int _MR = mc % MR;

    int i, j, I;

    assert(mc<=MC);
    assert(kc<=KC);

    for (I=0; I<MP; ++I) {
        pack_MRxk(kc, A, incRowA, incColA, buffer);
        A      += MR;
        buffer += MR*kc;
    }
    if (_MR>0) {
        for (j=0; j<kc; ++j) {
            for (i=0; i<_MR; ++i) {
                buffer[i] = A[i];
            }
            for (i=_MR; i<MR; ++i) {
                buffer[i] = 0.0;
            }
            A      += incColA;
            buffer += MR;
        }
    }
}

int
main()
{
    int i, j, mc, kc;

    initMatrix(M, K, A, M, 1);

    printf("A = \n");
    printMatrix(M, K, A, M);


    for (j=0; j<K; j+=KC) {
        kc = (j+KC<=K) ? KC : K - j;

        for (i=0; i<M; i+=MC) {
            mc = (i+MC<=M) ? MC : M - i;

            printf("Packing A(%d:%d, %d:%d)\n", i+1, i+mc, j+1, j+kc);
            pack_A(mc, kc, &A[i+j*M], 1, M, _A);

            printf("Buffer: _A = \n");
            printMatrix(MC, KC, _A, MC);
        }
    }
    return 0;
}
